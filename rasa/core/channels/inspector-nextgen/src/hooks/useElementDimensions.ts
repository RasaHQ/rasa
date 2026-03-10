import { useEffect } from "react";

export const useElementDimensionsEffect = (
  callback: (element: HTMLElement | null) => void,
  ref?: React.RefObject<HTMLElement | null>,
) => {
  useEffect(() => {
    const element = ref?.current;
    if (!element) return;

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          callback(ref?.current);
        }
      }
    });

    resizeObserver.observe(element);

    return () => {
      resizeObserver.disconnect();
    };
  }, [ref, callback]);

  return ref;
};

