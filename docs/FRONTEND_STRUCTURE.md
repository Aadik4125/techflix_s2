# Frontend Structure

Production frontend files (this is the whole runtime frontend — no other files are loaded):

- `frontend/index.html`: full page markup
- `frontend/styles/main.css`: consolidated styling
- `frontend/scripts/app.js`: main application logic
- `frontend/scripts/background.js`: Three.js background animation

`frontend/` previously also carried a second, unused componentized layout (`frontend/components/*.html` plus `api.js`, `recording.js`, `ui.js`, `component-loader.js`). Nothing in `index.html` loaded `component-loader.js`, so none of it ever ran. It has been moved to archive for reference:

- `archive/frontend/components/` (unused component partials)
- `archive/frontend/scripts/` (`api.js`, `recording.js`, `ui.js`, `component-loader.js`)
- `archive/frontend/test_1.html`
- `archive/frontend/new_frontend.txt`
