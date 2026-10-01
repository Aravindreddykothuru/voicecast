// Referenced by vitest.config.ts. Without this file the whole test run
// fails before collecting anything, which is how the frontend job stayed
// red on its first CI run.
import "@testing-library/jest-dom/vitest";

// jsdom implements neither of these, and the studio shell uses both: the
// responsive layout asks matchMedia which nav to render, and Framer Motion's
// layout animations measure elements. Stubbing them here keeps component
// tests about behaviour rather than about jsdom's gaps.
if (typeof window !== "undefined" && typeof window.matchMedia !== "function") {
  window.matchMedia = ((query: string) => ({
    matches: false, // desktop by default, matching the hook's own fallback
    media: query,
    onchange: null,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
  })) as unknown as typeof window.matchMedia;
}

if (typeof window !== "undefined" && !("ResizeObserver" in window)) {
  class RO {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  (window as unknown as { ResizeObserver: unknown }).ResizeObserver = RO;
}
