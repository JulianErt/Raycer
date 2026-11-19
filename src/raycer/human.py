"""Human viewer and keyboard play loop with pygame (API-15, API-18, REND-11, HUM-01..05)."""

import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from raycer._png import write_png
from raycer.env import RaycerEnv
from raycer.game import DISCRETE_ACTIONS
from raycer.palette import LABEL_COLORS
from raycer.physics import CarState
from raycer.recording import load_episode
from raycer.render import RenderOutput
from raycer.track import Track

INSTALL_HINT = 'pygame is not installed; install it with: uv add "raycer[human]"'

try:
    import pygame
except ImportError as e:
    raise ImportError(INSTALL_HINT) from e

VIEWS = ("rgb", "depth", "labels", "all")
_FPS = 30
COUNTDOWN_FRAMES = 3 * _FPS
_STEER_RISE_S, _STEER_RELEASE_S, _THROTTLE_S = 0.25, 0.15, 0.2
HELP_LINES = (
    "HELP",
    "W/A/S/D or arrows: drive",
    "R: restart",
    "P: pause",
    "M: minimap",
    "H/F1: help",
    "Tab: view",
    "F11: fullscreen",
    "F12: screenshot",
    "Esc: quit",
)
_KEYS_UP = (pygame.K_UP, pygame.K_w)
_KEYS_DOWN = (pygame.K_DOWN, pygame.K_s)
_KEYS_LEFT = (pygame.K_LEFT, pygame.K_a)
_KEYS_RIGHT = (pygame.K_RIGHT, pygame.K_d)


def keys_to_action(up: bool, down: bool, left: bool, right: bool) -> int:
    """Map held arrow keys to the API-07 discrete action index."""
    throttle = float(up) - float(down)
    steer = float(left) - float(right)
    return DISCRETE_ACTIONS.index((steer, throttle))


def start_scale(screen_w: int, screen_h: int, width: int, height: int) -> int:
    """Return the largest integer scale fitting 80 % of the screen, at least 2 (HUM-01)."""
    return max(2, min(math.floor(0.8 * screen_w / width), math.floor(0.8 * screen_h / height)))


def _approach(value: float, target: float, seconds: float, dt: float) -> float:
    """Move value towards target at full range per seconds, without overshoot."""
    step = dt / seconds
    if abs(target - value) <= step + 1e-9:
        return target
    return value + math.copysign(step, target - value)


def ramp_controls(
    steer: float, throttle: float, up: bool, down: bool, left: bool, right: bool, dt: float
) -> tuple[float, float]:
    """Advance the ramped (steer, throttle) state by dt for the held keys (HUM-05)."""
    steer_target = float(left) - float(right)
    steer_s = _STEER_RISE_S if steer_target else _STEER_RELEASE_S
    steer = _approach(steer, steer_target, steer_s, dt)
    throttle = _approach(throttle, float(up) - float(down), _THROTTLE_S, dt)
    return steer, throttle


def _fmt_time(seconds: float) -> str:
    """Format seconds as m:ss.ss."""
    return f"{int(seconds // 60)}:{seconds % 60:05.2f}"


def hud_lines(game_variables: NDArray[np.float32], info: dict[str, Any], laps: int) -> list[str]:
    """Return the HUD text: km/h first, then lap, lap times and WRONG WAY (HUM-02)."""
    lap_times: list[float] = info["lap_times"]
    lines = [
        f"{round(float(game_variables[0]) * 3.6)} km/h",
        f"lap {min(int(game_variables[2]) + 1, laps)}/{laps}",
        f"time {_fmt_time(float(game_variables[7]) - sum(lap_times))}",
        f"last {_fmt_time(lap_times[-1]) if lap_times else '--'}",
        f"best {_fmt_time(min(lap_times)) if lap_times else '--'}",
    ]
    if game_variables[5]:
        lines.append("WRONG WAY")
    return lines


class Viewer:
    """Resizable window that scales the RenderOutput it is given by an integer factor.

    HUD data, minimap and overlays are set by the Player and drawn only into the window.
    """

    def __init__(self, width: int, height: int, max_view_distance: float) -> None:
        """Open the window at the largest integer scale fitting 80 % of the screen."""
        pygame.display.init()
        pygame.font.init()
        self.width, self.height = width, height
        self.max_view_distance = max_view_distance
        info = pygame.display.Info()
        k = start_scale(info.current_w, info.current_h, width, height)
        self._windowed = (k * width, k * height)
        self.surface = pygame.display.set_mode(self._windowed, pygame.RESIZABLE)
        pygame.display.set_caption("Raycer")
        self.view = "rgb"
        self.fullscreen = False
        self.minimap = True
        self.hud: list[str] | None = None
        self.progress = 0.0
        self.track: Track | None = None
        self.car: CarState | None = None
        self.overlay: list[str] = []
        self._output: RenderOutput | None = None
        self._clock = pygame.time.Clock()
        self._font = pygame.font.Font(None, 20)
        self._big_font = pygame.font.Font(None, 64)

    def resize(self, size: tuple[int, int]) -> None:
        """Apply a window resize unless fullscreen."""
        if not self.fullscreen:
            self._windowed = size
            self.surface = pygame.display.set_mode(size, pygame.RESIZABLE)

    def toggle_fullscreen(self) -> None:
        """Switch between fullscreen and the previous windowed size."""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.surface = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.surface = pygame.display.set_mode(self._windowed, pygame.RESIZABLE)

    def _buffers(self) -> dict[str, NDArray[np.uint8] | None]:
        """Return the last output's buffers as (H, W, 3) uint8 images."""
        out = self._output
        depth = labels = None
        if out is not None and out.depth is not None:
            grey = np.clip(np.round(255 * (1 - out.depth / self.max_view_distance)), 0, 255)
            depth = np.repeat(grey.astype(np.uint8)[..., None], 3, axis=2)
        if out is not None and out.labels is not None:
            labels = LABEL_COLORS[out.labels]
        return {"rgb": None if out is None else out.rgb, "depth": depth, "labels": labels}

    def show(self, output: RenderOutput, hud: str) -> None:
        """Draw the scaled frame, HUD, minimap and overlay, capped at 30 fps."""
        self._output = output
        images = self._buffers()
        self.surface = pygame.display.get_surface()
        self.surface.fill((0, 0, 0))
        base: pygame.Surface | None = None
        if self.view == "all":
            base = pygame.Surface((2 * self.width, 2 * self.height))
            cells = {"rgb": (0, 0), "depth": (self.width, 0), "labels": (0, self.height)}
            for name, pos in cells.items():
                image = images[name]
                if image is not None:
                    base.blit(pygame.surfarray.make_surface(image.swapaxes(0, 1)), pos)
        elif (image := images[self.view]) is not None:
            base = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        if base is not None:
            (bw, bh), (sw, sh) = base.get_size(), self.surface.get_size()
            k = max(1, min(sw // bw, sh // bh))
            scaled = pygame.transform.scale(base, (k * bw, k * bh))
            self.surface.blit(scaled, ((sw - k * bw) // 2, (sh - k * bh) // 2))
        if self.hud is None:
            for i, line in enumerate(hud.splitlines()):
                text = self._font.render(line, True, (255, 255, 255), (0, 0, 0))
                self.surface.blit(text, (4, 4 + 18 * i))
        else:
            self._draw_hud(self.hud)
        if self.minimap and self.track is not None and self.car is not None:
            self._draw_minimap(self.track, self.car)
        if self.overlay:
            self._draw_overlay(self.overlay)
        pygame.display.flip()
        pygame.event.pump()
        self._clock.tick(_FPS)

    def _draw_hud(self, lines: list[str]) -> None:
        """Draw the HUD panel: big speed, text lines, lap progress bar, WRONG WAY."""
        texts = [self._big_font.render(lines[0], True, (255, 255, 255))]
        for line in lines[1:]:
            if line == "WRONG WAY":
                texts.append(self._big_font.render(line, True, (255, 40, 40)))
            else:
                texts.append(self._font.render(line, True, (255, 255, 255)))
        bar_w = 120
        w = max(bar_w, *(t.get_width() for t in texts)) + 16
        h = sum(t.get_height() for t in texts) + 30
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 150))
        y = 8
        for text in texts:
            panel.blit(text, (8, y))
            y += text.get_height()
        y += 6
        pygame.draw.rect(panel, (255, 255, 255), (8, y, bar_w, 10), 1)
        fill = round(bar_w * min(max(self.progress, 0.0), 1.0))
        pygame.draw.rect(panel, (80, 220, 80), (8, y, fill, 10))
        self.surface.blit(panel, (8, 8))

    def _draw_minimap(self, track: Track, car: CarState) -> None:
        """Draw walls, finish line and the car arrow in the top-right corner (HUM-03)."""
        sw, sh = self.surface.get_size()
        size = max(min(sw, sh) // 4, 40)
        points = np.concatenate([track.seg_a, track.seg_b])
        lo, hi = points.min(axis=0), points.max(axis=0)
        scale = (size - 12) / float(max(hi - lo))

        def px(x: float, y: float) -> tuple[float, float]:
            """Map world metres to panel pixels, y up."""
            return 6 + (x - lo[0]) * scale, size - 6 - (y - lo[1]) * scale

        panel = pygame.Surface((size, size), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 150))
        for a, b in zip(track.seg_a, track.seg_b, strict=True):
            pygame.draw.line(panel, (200, 200, 200), px(*a), px(*b))
        pygame.draw.polygon(panel, (255, 220, 0), [px(*c) for c in track.finish_quad])
        c, s, r = math.cos(car.heading), math.sin(car.heading), 6 / scale
        arrow = [
            (car.x + 1.5 * r * c, car.y + 1.5 * r * s),
            (car.x - r * c - r * s, car.y - r * s + r * c),
            (car.x - r * c + r * s, car.y - r * s - r * c),
        ]
        pygame.draw.polygon(panel, (255, 60, 60), [px(*p) for p in arrow])
        self.surface.blit(panel, (sw - size - 8, 8))

    def _draw_overlay(self, lines: list[str]) -> None:
        """Dim the window and draw the first line large, the rest small, centred."""
        sw, sh = self.surface.get_size()
        dim = pygame.Surface((sw, sh), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 120))
        self.surface.blit(dim, (0, 0))
        texts = [self._big_font.render(lines[0], True, (255, 255, 255))]
        texts += [self._font.render(line, True, (255, 255, 255)) for line in lines[1:]]
        y = (sh - sum(t.get_height() for t in texts)) // 2
        for text in texts:
            self.surface.blit(text, ((sw - text.get_width()) // 2, y))
            y += text.get_height()

    def cycle_view(self) -> None:
        """Advance RGB -> depth -> labels -> all, skipping views without a buffer."""
        images = self._buffers()
        i = VIEWS.index(self.view)
        while True:
            i = (i + 1) % len(VIEWS)
            if VIEWS[i] == "all" or images[VIEWS[i]] is not None:
                break
        self.view = VIEWS[i]

    def screenshot(self, path: str | os.PathLike[str]) -> None:
        """Write the current window contents as PNG."""
        write_png(path, pygame.surfarray.array3d(self.surface).swapaxes(0, 1))

    def close(self) -> None:
        """Close the window."""
        pygame.display.quit()


class Player:
    """Keyboard play loop over a human-mode env (or a wrapper around one)."""

    def __init__(self, env: Any, seed: int | None = None) -> None:
        """Reset the env with seed and start with no keys held."""
        self.env = env
        self.viewer: Viewer = env.unwrapped.viewer
        self.paused = False
        self.help = False
        self._continuous = isinstance(env.action_space, spaces.Box)
        self._held: set[int] = set()
        self._reset(seed)

    def _reset(self, seed: int | None = None) -> None:
        """Reset the env, controls and finish state and start the countdown."""
        _, self.info = self.env.reset(seed=seed)
        self.countdown = COUNTDOWN_FRAMES
        self.finished = False
        self.controls = (0.0, 0.0)
        self.viewer.track = self.env.unwrapped.game.get_track()

    def action(self) -> Any:
        """Return the env action for the held keys, advancing the ramp in continuous mode."""
        keys = (_KEYS_UP, _KEYS_DOWN, _KEYS_LEFT, _KEYS_RIGHT)
        up, down, left, right = (not self._held.isdisjoint(k) for k in keys)
        if not self._continuous:
            return keys_to_action(up, down, left, right)
        self.controls = ramp_controls(*self.controls, up, down, left, right, 1 / _FPS)
        return np.array(self.controls, np.float32)

    def _overlay(self) -> list[str]:
        """Return the overlay lines for the current state."""
        if self.help:
            return list(HELP_LINES)
        if self.paused:
            return ["PAUSE"]
        if self.finished:
            times = [f"lap {i}: {_fmt_time(t)}" for i, t in enumerate(self.info["lap_times"], 1)]
            return ["FINISH", *times, "R: restart"]
        if self.countdown > 0:
            return [str(math.ceil(self.countdown / _FPS))]
        return []

    def frame(self) -> bool:
        """Handle pending events, step unless frozen and render; False once the user quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.VIDEORESIZE:
                self.viewer.resize(event.size)
            elif event.type == pygame.KEYUP:
                self._held.discard(event.key)
            elif event.type == pygame.KEYDOWN:
                self._held.add(event.key)
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_r:
                    self._reset()
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                elif event.key in (pygame.K_h, pygame.K_F1):
                    self.help = not self.help
                elif event.key == pygame.K_m:
                    self.viewer.minimap = not self.viewer.minimap
                elif event.key == pygame.K_TAB:
                    self.viewer.cycle_view()
                elif event.key == pygame.K_F11:
                    self.viewer.toggle_fullscreen()
                elif event.key == pygame.K_F12:
                    name = f"raycer_{datetime.now():%Y%m%d-%H%M%S}.png"
                    self.viewer.screenshot(Path.cwd() / name)
        if self.paused or self.help or self.finished:
            pass  # physics frozen
        elif self.countdown > 0:
            self.countdown -= 1
        else:
            _, _, terminated, truncated, self.info = self.env.step(self.action())
            self.finished = terminated
            if truncated:
                self._reset()
        game = self.env.unwrapped.game
        gv = game._snapshot().game_variables
        self.viewer.hud = hud_lines(gv, self.info, game._laps)
        self.viewer.progress = float(gv[1])
        self.viewer.car = game._car
        self.viewer.overlay = self._overlay()
        self.env.render()
        return True


def play(env: Any, seed: int | None = None) -> None:
    """Run the play loop until the user quits."""
    player = Player(env, seed)
    while player.frame():
        pass


def replay(path: str | Path) -> list[NDArray[np.uint8]]:
    """Step a recorded episode's actions at 30 Hz and return the screens it produced."""
    ep = load_episode(path)
    meta = ep["meta"]
    env = RaycerEnv(
        meta["scenario"],
        render_mode="human",
        action_type=meta["action_type"],
        width=meta["width"],
        height=meta["height"],
        frame_skip=meta["frame_skip"],
        depth="depth" in ep,
        labels="labels" in ep,
    )
    try:
        options = None if meta["track_seed"] is None else {"track_seed": meta["track_seed"]}
        obs, _ = env.reset(seed=meta["seed"], options=options)
        screens = [obs["screen"]]
        env.render()
        for action in ep["actions"]:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    return screens
            screens.append(env.step(action)[0]["screen"])
            env.render()
        return screens
    finally:
        env.close()
