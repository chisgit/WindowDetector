# Project Rules

- Make detector fixes in code, not by hardcoding page, room, group-home, or coordinate-specific overrides.
- Treat user corrections as evidence for reusable computer-vision rules that should work across many diagrams.
- Expect multiple window symbol families over time. Keep detector logic modular so new symbol types can be added without breaking existing ones.
- Door openings, door swings, wall gaps, labels, dimensions, fixtures, and spaces between a door and a window cap are not windows.
- When a measurement or dimension line passes through a window symbol, prefer rules that recover the underlying cap/pane evidence instead of relying on uninterrupted strokes.
