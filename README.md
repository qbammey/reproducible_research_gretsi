Code for the slides of the Gretsi 2025 tutorial ititle: “Reproducible Research: From Code to Career”.

Render with Quarto: `uv run quarto render index.qmd`, then serve the resulting html file (e. g. `python -m http.server`), or directly preview with `uv run quarto preview index.qmd`.

To run the demos: `uv run streamlit run example_demo/examples/outlier_colour_transfer/{app.py, app_full.py}`

Credits:

* Slides: Quentin Bammey, Gabriele Facciolo
* Example demo code:
  * Code: original authors of the Gretsi article, “Un algorithme de point fixe pour calculer des barycentres robustes entre mesures”, Eloi Tanguy, Julie Delon, Nathaël Gozan, Gretsi 2025
  * Demos (example_demo/examples/outlier_colour_transfer/{app.py, app_full.py}): Quentin Bammey
