## AI Material Classification

Frontend-only app for intelligent waste material classification using Google Gemini.

### Local development

```bash
npm install
npm run dev
```

Open http://localhost:5000

### Build

```bash
npm run build
npm run preview
```

### GitHub Pages

1. Push this repo to GitHub.
2. In repo settings, set **Pages** source to **GitHub Actions**.
3. Push to `main` or `master` to trigger deployment.

The app runs entirely in the browser. After uploading an image, users enter their own Gemini API key, which is stored locally in the browser.

Optional environment variable:

```bash
VITE_GEMINI_MODEL=gemini-3.1-flash-lite
```

Get an API key from Google AI Studio: https://aistudio.google.com/apikey
