/** Ambient typing for plotly.js-dist-min (pre-built bundle). */

declare module 'plotly.js-dist-min' {
  const Plotly: {
    newPlot: (element: HTMLElement, data: unknown[], layout?: Record<string, unknown>, config?: Record<string, unknown>) => Promise<unknown>;
    react: (element: HTMLElement, data: unknown[], layout?: Record<string, unknown>, config?: Record<string, unknown>) => Promise<unknown>;
    purge: (element: HTMLElement) => void;
  };
  export default Plotly;
}
