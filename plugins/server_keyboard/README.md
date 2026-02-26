# Keyboard webpolicy server

Serves keyboard presses over the webpolicy API.

## Run

```bash
python plugins/server_keyboard/keyboard.py --host 0.0.0.0 --port 8080
```

Returns key presses as a dict of booleans, e.g.:

```python
{"w": True, "up": True}
```
