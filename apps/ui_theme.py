"""
YAML-configurable curses style definitions for the ham2mon UI.

A future CLI flag will let a user point at their own theme file instead of
the bundled one (ThemeConfiguration.file_name exists for that); until then,
the bundled DEFAULT_THEME_FILE is always what's loaded.

Usage (see setup_screen() in cursesgui.py for the real integration):

    theme_config = ThemeConfiguration(file_name=args.theme_config)  # may be None -> bundled file
    THEME.config = theme_config
    THEME.load()                       # reads and validates the theme file; no curses calls yet
    # ... curses.initscr(); curses.start_color() must happen before resolve() ...
    THEME.resolve()                    # does all init_pair()/color_pair() work
    # ... THEME.get("channel.border") from here on, in any draw path ...
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict, cast, final

import curses
import logging

import yaml

logger: Final[logging.Logger] = logging.getLogger(f"ham2mon.{__name__}")


class StyleSpec(TypedDict, total=False):
    """One style entry as it appears in DEFAULT_STYLES / a style YAML file.

    fg/bg accept either one of the 8 named colors below, or an int 0-255
    xterm-256 color index. Numeric colors are degraded to the nearest
    named color at resolve() time on terminals that don't report 256-color
    support -- see _nearest_basic_color().
    """
    fg: str | int
    bg: str | int
    bold: bool
    dim: bool
    italic: bool
    underline: bool
    reverse: bool
    blink: bool
    standout: bool


_COLOR_MAP: Final[dict[str, int]] = {
    'black': curses.COLOR_BLACK,
    'red': curses.COLOR_RED,
    'green': curses.COLOR_GREEN,
    'yellow': curses.COLOR_YELLOW,
    'blue': curses.COLOR_BLUE,
    'magenta': curses.COLOR_MAGENTA,
    'cyan': curses.COLOR_CYAN,
    'white': curses.COLOR_WHITE,
}

def _resolve_color(raw: str | int) -> int:
    """Resolves a validated fg/bg value (already known to be either a
    known color name or an in-range int, per _parse_style_spec) to a
    concrete curses color constant."""
    if isinstance(raw, str):
        return _COLOR_MAP.get(raw, curses.COLOR_WHITE)
    return raw

# style-file key -> curses attribute constant name (looked up via getattr so
# we degrade gracefully on terminals/builds missing a given attribute, e.g.
# A_ITALIC on some platforms, instead of raising).
_FLAG_NAMES: Final[tuple[str, ...]] = (
    'bold', 'dim', 'italic', 'underline', 'reverse', 'blink', 'standout',
)
_FLAG_ATTR_NAMES: Final[dict[str, str]] = {
    'bold': 'A_BOLD',
    'dim': 'A_DIM',
    'italic': 'A_ITALIC',
    'underline': 'A_UNDERLINE',
    'reverse': 'A_REVERSE',
    'blink': 'A_BLINK',
    'standout': 'A_STANDOUT',
}

# Some attributes exist as curses.A_* constants on every platform but only
# actually render if the terminal's terminfo entry declares the matching
# capability (e.g. italic needs "sitm"/"ritm"). Checking hasattr(curses,
# 'A_ITALIC') alone (as this module used to) only proves Python's curses
# module knows the constant -- it says nothing about whether the terminal
# will honor it, so a mismatch silently no-ops instead of erroring. This
# maps flag name -> terminfo capability name so resolve() can check the
# terminal's actual advertised support and warn instead of guessing.
_FLAG_TERMINFO_CAPS: Final[dict[str, str]] = {
    'italic': 'sitm',
    'underline': 'smul',
    'standout': 'smso',
    'blink': 'blink',
}
# 'bold', 'dim', and 'reverse' aren't checked: they're supported almost
# universally, and reverse-video in particular has no single clean terminfo
# capability name to test (it's synthesized from colors on many terminals).

_KNOWN_SPEC_KEYS: Final[frozenset[str]] = frozenset({'fg', 'bg', *_FLAG_NAMES})

# Fallback values for a single malformed/missing entry *within* an otherwise
# valid, present default-theme.yaml (e.g. a typo'd color name for one style key).
# This is NOT a substitute for the file itself -- default-theme.yaml is a required
# application resource (see module docstring) and ThemeManager.load() raises
# if it's missing entirely. These per-key defaults exist purely so one bad
# line in an otherwise-good file degrades gracefully instead of crashing the
# whole UI, and they document today's shipped colors for reference.
DEFAULT_STYLES: Final[dict[str, StyleSpec]] = {
    'screen.border':                   {'fg': 'white'},

    'spectrum.border':                 {'fg': 'white'},
    'spectrum.title':                  {'fg': 'white', 'dim': True, 'bold': True},
    'spectrum.bar_above_max':          {'fg': 'red', 'bold': True},
    'spectrum.bar_between':            {'fg': 'green', 'bold': True},
    'spectrum.bar_below_threshold':    {'fg': 'cyan', 'bold': True},
    'spectrum.max_db_label':           {'fg': 'red'},
    'spectrum.min_db_label':           {'fg': 'cyan'},
    'spectrum.threshold_line':         {'fg': 'green'},
    'spectrum.threshold_label':        {'fg': 'green'},
    'spectrum.channel_marker_active':  {'fg': 'blue'},
    'spectrum.channel_marker_hanging': {'fg': 'blue', 'dim': True},

    'channel.border':                  {'fg': 'white'},
    'channel.title':                   {'fg': 'white', 'dim': True, 'bold': True},
    'channel.freq_active':             {'fg': 'green', 'bold': True},
    'channel.icon_active':             {'fg': 'green'},
    'channel.freq_inactive':           {'fg': 'green', 'dim': True},
    'channel.icon_inactive':           {'fg': 'green', 'dim': True},
    'channel.index_active':            {'fg': 'blue'},
    'channel.index_inactive':          {'fg': 'blue', 'dim': True},
    'channel.placeholder_index':       {'fg': 'blue', 'dim': True},
    'channel.placeholder_text':        {'fg': 'white', 'dim': True},

    'lockout.border':                  {'fg': 'white'},
    'lockout.title':                   {'fg': 'white', 'dim': True, 'bold': True},
    'lockout.active':                  {'fg': 'yellow', 'bold': True},
    'lockout.inactive':                {'fg': 'white'},

    'receiver.border':                 {'fg': 'white'},
    'receiver.title':                  {'fg': 'white', 'dim': True, 'bold': True},
    'receiver.value_editable':         {'fg': 'yellow'},
    'receiver.value_readonly':         {'fg': 'white'},
}

# The style file ships in the same directory as this module (and the rest of
# ham2mon's application files) -- resolved via __file__ so it's found
# correctly no matter what directory the process was launched from. This is
# the file ThemeManager loads unless a future --theme-config CLI flag
# supplies an explicit override path via ThemeConfiguration.file_name.
DEFAULT_THEME_FILE: Final[Path] = Path(__file__).resolve().parent / 'default-theme.yaml'


@final
@dataclass(kw_only=True)
class ThemeConfiguration:
    """File_name only exists so a future CLI flag can point at a *different*,
    user-supplied theme file instead.
    """
    file_name: Path | None = None


def _as_dict_of_str_keys(value: object) -> dict[str, object] | None:
    """Narrow an untyped YAML node to dict[str, object], or None if it isn't
    a mapping with all-string keys. Keeps every value coming out of
    yaml.safe_load() explicitly checked rather than trusted."""
    if not isinstance(value, dict):
        return None
    mapping = cast('dict[object, object]', value)
    result: dict[str, object] = {}
    for raw_key, raw_val in mapping.items():
        if not isinstance(raw_key, str):
            return None
        result[raw_key] = raw_val
    return result


def _validate_color(source: Path, name: str, key: str, raw: object) -> str | int | None:
    """Validate one fg/bg value: either a known color name, or an int
    0-255 xterm-256 index. Returns the validated value, or None (with a
    logged warning) if it's neither.

    bool is deliberately rejected even though it's a subclass of int in
    Python (isinstance(True, int) is True) -- a stray `fg: true` in YAML
    should not silently resolve to color index 1.
    """
    if isinstance(raw, str):
        if raw in _COLOR_MAP:
            return raw
        logger.warning(
            f'{source}: style "{name}" has unknown {key} color {raw!r} ' +
            f'(expected one of {sorted(_COLOR_MAP)}, or an int 0-255); ignoring entry')
        return None

    if isinstance(raw, int) and not isinstance(raw, bool):
        if 0 <= raw <= 255:
            return raw
        logger.warning(
            f'{source}: style "{name}" has out-of-range {key} color {raw} ' +
            '(expected 0-255); ignoring entry')
        return None

    logger.warning(
        f'{source}: style "{name}" has invalid {key} color {raw!r} ' +
        f'(expected one of {sorted(_COLOR_MAP)}, or an int 0-255); ignoring entry')
    return None


def _parse_style_spec(source: Path, name: str, value: object) -> StyleSpec | None:
    """Validate one style entry from a YAML file. Returns None (with a
    logged warning) if the entry is unusable; unknown/invalid attributes
    within an otherwise-valid entry are dropped individually."""
    node = _as_dict_of_str_keys(value)
    if node is None:
        logger.warning(f'{source}: style "{name}" must be a mapping; ignoring')
        return None

    fg_raw = _validate_color(source, name, 'fg', node.get('fg', 'white'))
    if fg_raw is None:
        return None

    bg_raw = _validate_color(source, name, 'bg', node.get('bg', 'black'))
    if bg_raw is None:
        return None

    spec: StyleSpec = {'fg': fg_raw, 'bg': bg_raw}

    bold_raw = node.get('bold')
    if isinstance(bold_raw, bool):
        spec['bold'] = bold_raw
    dim_raw = node.get('dim')
    if isinstance(dim_raw, bool):
        spec['dim'] = dim_raw
    italic_raw = node.get('italic')
    if isinstance(italic_raw, bool):
        spec['italic'] = italic_raw
    underline_raw = node.get('underline')
    if isinstance(underline_raw, bool):
        spec['underline'] = underline_raw
    reverse_raw = node.get('reverse')
    if isinstance(reverse_raw, bool):
        spec['reverse'] = reverse_raw
    blink_raw = node.get('blink')
    if isinstance(blink_raw, bool):
        spec['blink'] = blink_raw
    standout_raw = node.get('standout')
    if isinstance(standout_raw, bool):
        spec['standout'] = standout_raw

    for key in node:
        if key not in _KNOWN_SPEC_KEYS:
            logger.warning(f'{source}: style "{name}" has unknown attribute "{key}"; ignoring attribute')

    return spec


def _get_flag(spec: StyleSpec, flag_name: str) -> bool:
    """Read one boolean flag out of a StyleSpec by name (flag_name is one of
    _FLAG_NAMES). A small typed indirection so callers don't need a literal
    key, which TypedDict item access otherwise requires."""
    if flag_name == 'bold':
        return spec.get('bold', False)
    if flag_name == 'dim':
        return spec.get('dim', False)
    if flag_name == 'italic':
        return spec.get('italic', False)
    if flag_name == 'underline':
        return spec.get('underline', False)
    if flag_name == 'reverse':
        return spec.get('reverse', False)
    if flag_name == 'blink':
        return spec.get('blink', False)
    if flag_name == 'standout':
        return spec.get('standout', False)
    return False


def _terminal_declares_capability(cap_name: str) -> bool:
    """Ask the terminal's terminfo entry (not just the Python curses module)
    whether it actually advertises support for a capability like "sitm"
    (start italics). Requires curses.setupterm() to have run, which
    curses.initscr() already does before StyleManager.resolve() is called.
    Returns False (rather than raising) if the check itself can't run --
    e.g. no terminal attached -- so a broken capability query degrades to
    "assume unsupported" instead of crashing the whole UI at startup.
    """
    try:
        return curses.tigetstr(cap_name) is not None
    except curses.error:
        return False


@final
class ThemeManager:
    """Loads a human-readable style config and resolves it into curses
    attribute ints exactly once, at startup.
    """

    def __init__(self, config: ThemeConfiguration | None = None) -> None:
        self.config: ThemeConfiguration = config if config is not None else ThemeConfiguration()
        self._raw: dict[str, StyleSpec] = {name: StyleSpec(**spec) for name, spec in DEFAULT_STYLES.items()}
        self._resolved: dict[str, int] = {}
        self._loaded: bool = False
        self._resolved_flag: bool = False

    def load(self) -> None:
        """Read and validate the theme file (the bundled DEFAULT_THEME_FILE,
        or the override in self.config.file_name if one was supplied).

        Individual malformed/missing entries *within* an existing, valid
        file still degrade gracefully to DEFAULT_STYLES for that one key
        (see DEFAULT_STYLES' docstring) -- only the file's presence is
        strict.
        """
        self._loaded = True

        file = self.config.file_name if self.config.file_name is not None else DEFAULT_THEME_FILE

        if not file.exists():
            raise FileNotFoundError(
                f'Required application resource is missing: {file}. ' +
                'default-theme.yaml ships with ham2mon and must be present ' +
                'alongside the application; if it was deleted, reinstall ' +
                'or restore it from the repository.')

        logger.debug(f'Loading UI styles from {file}')
        with file.open(mode='r') as fh:
            try:
                loaded: object = yaml.safe_load(fh)  # pyright: ignore[reportAny] -- yaml.safe_load's stub returns Any; narrowed immediately below via _as_dict_of_str_keys
            except yaml.YAMLError as e:
                if isinstance(e, yaml.MarkedYAMLError):
                    logger.error(f'{e.problem_mark} {e.problem} {e.context if e.context else ""}')
                else:
                    logger.error(f'Something went wrong while parsing yaml file: {file}')
                raise Exception(
                    "Invalid yaml style file (enable debugging for more info)") from e

        data = _as_dict_of_str_keys(loaded)
        if data is None:
            logger.warning(f'{file}: top-level content must be a mapping; file is present but malformed, using per-key defaults')
            return

        styles_node = data.get('styles')
        if styles_node is None:
            logger.warning(f'{file} has no top-level "styles" key; file is present but malformed, using per-key defaults')
            return

        overrides = _as_dict_of_str_keys(styles_node)
        if overrides is None:
            logger.warning(f'{file}: "styles" must be a mapping; ignoring entire file, using per-key defaults')
            return

        for name, raw_spec in overrides.items():
            if name not in DEFAULT_STYLES:
                logger.warning(
                    f'{file}: unknown style key "{name}" (no matching UI element); ignoring')
                continue
            spec = _parse_style_spec(file, name, raw_spec)
            if spec is not None:
                self._raw[name] = spec

    def resolve(self) -> None:
        """Do all curses.init_pair()/color_pair() work. Call exactly once,
        after curses.start_color(). Safe to call even when the terminal has
        no color support (falls back to attribute-only styling, e.g. bold).
        """
        if not self._loaded:
            self.load()

        has_color: bool = curses.has_colors()
        max_pairs: int = curses.COLOR_PAIRS if has_color else 0

        # curses.COLORS is populated by curses.start_color(), which every
        # caller of resolve() already invokes first (see module docstring).
        has_256: bool = has_color and curses.COLORS >= 256
        if not has_256 and has_color:
            uses_numeric_color = any(
                isinstance(spec.get('fg'), int) or isinstance(spec.get('bg'), int)
                for spec in self._raw.values())
            if uses_numeric_color:
                logger.warning(
                    f'Terminal reports {curses.COLORS} colors (need 256); falling back ' +
                    'all styles to DEFAULT_STYLES for legibility. ' +
                    'Check $TERM (e.g. use "xterm-256color").')
                self._raw = {name: StyleSpec(**spec) for name, spec in DEFAULT_STYLES.items()}

        pair_index: dict[tuple[int, int], int] = {}
        next_pair = 1  # curses reserves pair 0 (default fg/bg)
        cap_support: dict[str, bool] = {}  # terminfo capability name -> supported, cached per resolve() call

        for name, spec in self._raw.items():
            attr: int = curses.A_NORMAL

            if has_color:
                fg = _resolve_color(spec.get('fg', 'white'))
                bg = _resolve_color(spec.get('bg', 'black'))
                key = (fg, bg)

                if key not in pair_index:
                    if next_pair >= max_pairs:
                        logger.warning(
                            f'Style "{name}" needs a new color pair but the terminal ' +
                            f'only supports {max_pairs}; reusing an existing pair')
                        pair_index[key] = 1 if max_pairs > 1 else 0
                    else:
                        curses.init_pair(next_pair, fg, bg)
                        pair_index[key] = next_pair
                        next_pair += 1

                if pair_index[key]:
                    attr |= curses.color_pair(pair_index[key])

            for flag_name in _FLAG_NAMES:
                if not _get_flag(spec, flag_name):
                    continue
                curses_attr_name = _FLAG_ATTR_NAMES[flag_name]
                curses_flag: int | None = getattr(curses, curses_attr_name, None)
                if curses_flag is None:
                    logger.warning(
                        f'Style "{name}" requests "{flag_name}", which this terminal/' +
                        'curses build does not support; ignoring')
                    continue

                cap_name = _FLAG_TERMINFO_CAPS.get(flag_name)
                if cap_name is not None:
                    if cap_name not in cap_support:
                        cap_support[cap_name] = _terminal_declares_capability(cap_name)
                    if not cap_support[cap_name]:
                        logger.warning(
                            f'Style "{name}" requests "{flag_name}", but this terminal\'s ' +
                            f'terminfo entry does not declare the "{cap_name}" capability, ' +
                            'so it will render as no-op; ignoring. Check $TERM and, if ' +
                            'you\'re inside tmux/screen, its passthrough settings.')
                        continue

                attr |= curses_flag

            self._resolved[name] = attr

        self._resolved_flag = True

    def get(self, name: str) -> int:
        """Dict lookup."""
        if not self._resolved_flag:
            logger.warning('ThemeManager.get() called before resolve(); returning A_NORMAL')
            return curses.A_NORMAL
        resolved = self._resolved.get(name)
        if resolved is None:
            logger.warning(f'Unknown style key "{name}"; using A_NORMAL')
            return curses.A_NORMAL
        return resolved


# Module-level singleton
THEME: Final[ThemeManager] = ThemeManager()
