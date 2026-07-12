# Static (Web Frontend)

This directory houses the HTML, CSS, and JavaScript that power the PeriodyX interactive web application.

## Architecture

- **`index.html`**: The main interface. Designed with a dark, glassmorphism aesthetic. Contains the configuration panels, the batch-upload zone, and the results dashboard.
- **`style.css`**: Implements the responsive grid, animations (e.g., the glowing orbit loader), and modern typography.
- **`app.js`**: The vanilla JavaScript client that communicates with the FastAPI backend. It handles file uploads, dynamically renders the batch results table, and updates the UI (including single-transit alerts and bootstrap uncertainty confidence bars) based on the backend JSON payloads.
