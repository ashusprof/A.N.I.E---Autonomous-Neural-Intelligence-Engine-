# A.N.I.E Frontend

Web interface for A.N.I.E - Autonomous Neural Intelligence Engine.

## Quick Start

```bash
# Using npm
npm run start

# Or directly with npx
npx http-server -p 3002 -o /fullscreen.html
```

Then open http://localhost:3002/fullscreen.html

## Files

| File | Description |
|------|-------------|
| `fullscreen.html` | Main HTML interface |
| `styles-fullscreen.css` | Main stylesheet |
| `app-fullscreen.js` | Chat application logic |
| `voice-mode.css` | Voice mode styles |
| `voice-mode-methods.js` | Voice mode utilities |
| `anie.png` | Application logo |

## Backend Connection

The frontend connects to the backend at `http://127.0.0.1:8000`.

To change this, edit the `CONFIG.BACKEND_URL` in `app-fullscreen.js`.
