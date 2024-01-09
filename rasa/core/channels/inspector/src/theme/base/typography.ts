import "./fonts/fontFaces.css";

export const fonts = {
  body: "Lato, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen-Sans, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif",
  heading:
    "Lato, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen-Sans, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif",
  mono: '"IBM Plex Mono", "Courier New", Courier, monospace',
};
export type RasaFonts = typeof fonts;

export const rasaFontSizes = {
  xs: "0.5rem",
  sm: "0.813rem",
  md: "0.875rem",
  lg: "1rem",
  xl: "1.5rem",
};
export type RasaFontSizes = typeof rasaFontSizes;

export const rasaFontWeights = {
  normal: 400,
  bold: 700,
};
export type RasaFontWeights = typeof rasaFontWeights;
