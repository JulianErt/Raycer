"""Human viewer and keyboard play loop with pygame (API-15, API-18, REND-11)."""

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
from raycer.recording import load_episode
from raycer.render import RenderOutput

INSTALL_HINT = 'pygame is not installed; install it with: uv add "raycer[human]"'

try:
    import pygame
except ImportError as e:
    raise ImportError(INSTALL_HINT) from e

VIEWS = ("rgb", "depth", "labels", "all")
_FPS = 30
_KEYS_UP = (pygame.K_UP, pygame.K_w)
_KEYS_DOWN = (pygame.K_DOWN, pygame.K_s)
_KEYS_LEFT = (pygame.K_LEFT, pygame.K_a)
_KEYS_RIGHT = (pygame.K_RIGHT, pygame.K_d)


def keys_to_action(up: bool, down: bool, left: bool, right: bool) -> int:
    """Map held arrow keys to the API-07 discrete action index."""
    throttle = float(up) - float(down)
    steer = float(left) - float(right)
    return DISCRETE_ACTIONS.index((steer, throttle))


class Viewer:
    """Window of 2W x 2H that draws only the RenderOutput it is given (REND-11)."""

    def __init__(self, width: int, height: int, max_view_distance: float) -> None:
        """Open the window."""
        pygame.display.init()
        pygame.font.init()
        self.width, self.height = width, height
        self.max_view_distance = max_view_distance
        self.surface = pygame.display.set_mode((2 * width, 2 * height))
        pygame.display.set_caption("Raycer")
        self.view = "rgb"
        self._output: RenderOutput | None = None
        self._clock = pygame.time.Clock()
        self._font = pygame.font.Font(None, 20)

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
        """Draw the frame and the HUD text, capped at 30 fps."""
        self._output = output
        images = self._buffers()
        self.surface.fill((0, 0, 0))
        if self.view == "all":
            cells = {"rgb": (0, 0), "depth": (self.width, 0), "labels": (0, self.height)}
            for name, pos in cells.items():
                image = images[name]
                if image is not None:
                    self.surface.blit(pygame.surfarray.make_surface(image.swapaxes(0, 1)), pos)
        else:
            image = images[self.view]
            if image is not None:
                surf = pygame.surfarray.make_surface(image.swapaxes(0, 1))
                self.surface.blit(pygame.transform.scale(surf, self.surface.get_size()), (0, 0))
        for i, line in enumerate(hud.splitlines()):
            text = self._font.render(line, True, (255, 255, 255), (0, 0, 0))
            self.surface.blit(text, (4, 4 + 18 * i))
        pygame.display.flip()
        pygame.event.pump()
        self._clock.tick(_FPS)

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
        self._continuous = isinstance(env.action_space, spaces.Box)
        self._held: set[int] = set()
        _, self.info = env.reset(seed=seed)

    def action(self) -> Any:
        """Return the env action for the currently held keys."""
        keys = (_KEYS_UP, _KEYS_DOWN, _KEYS_LEFT, _KEYS_RIGHT)
        idx = keys_to_action(*(not self._held.isdisjoint(k) for k in keys))
        return np.array(DISCRETE_ACTIONS[idx], np.float32) if self._continuous else idx

    def frame(self) -> bool:
        """Handle pending events, step unless paused and render; False once the user quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYUP:
                self._held.discard(event.key)
            elif event.type == pygame.KEYDOWN:
                self._held.add(event.key)
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_r:
                    _, self.info = self.env.reset()
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                elif event.key == pygame.K_TAB:
                    self.viewer.cycle_view()
                elif event.key == pygame.K_F12:
                    name = f"raycer_{datetime.now():%Y%m%d-%H%M%S}.png"
                    self.viewer.screenshot(Path.cwd() / name)
        if not self.paused:
            _, _, terminated, truncated, self.info = self.env.step(self.action())
            if terminated or truncated:
                _, self.info = self.env.reset()
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
