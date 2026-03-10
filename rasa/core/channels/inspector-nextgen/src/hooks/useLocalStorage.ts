import { useState, useEffect, useCallback, useRef } from "react";
import { useInspectorContext } from "../InspectorContext";

interface LocalStorageChangeEvent extends CustomEvent {
  detail: {
    key: string;
    newValue: string | null;
    oldValue: string | null;
  };
}

export function useLocalStorage<T>(
  key: string,
  initialValue: T,
): [T, (value: T | ((prev: T) => T)) => void, () => void] {
  const { logError } = useInspectorContext();
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      return readFromStorage(key, initialValue);
    } catch (error) {
      logError(error, {
        tags: { component: "useLocalStorage", action: "readFromStorage" },
        extra: { key },
      });
      return initialValue;
    }
  });

  // Stable reference to avoid effect re-runs
  const keyRef = useRef(key);
  const initialValueRef = useRef(initialValue);

  useEffect(() => {
    keyRef.current = key;
  }, [key]);

  useEffect(() => {
    initialValueRef.current = initialValue;
  }, [initialValue]);

  useEffect(() => {
    const handleLocalStorageChange = (event: LocalStorageChangeEvent) => {
      if (event.detail.key !== keyRef.current) return;

      const newValue: T = event.detail.newValue
        ? (JSON.parse(event.detail.newValue) as T)
        : initialValueRef.current;

      setStoredValue(newValue);
    };

    globalThis.addEventListener(
      "local-storage-changed",
      handleLocalStorageChange as EventListener,
    );

    return () => {
      globalThis.removeEventListener(
        "local-storage-changed",
        handleLocalStorageChange as EventListener,
      );
    };
  }, []); // Empty deps - refs handle updates + we keep one event listener per window

  const setValue = useCallback(
    (value: T | ((prev: T) => T)) => {
      const valueToStore =
        typeof value === "function"
          ? (value as (prev: T) => T)(storedValue)
          : value;

      try {
        const oldValue = localStorage.getItem(keyRef.current);
        const serializedValue = JSON.stringify(valueToStore);

        localStorage.setItem(keyRef.current, serializedValue);

        // Dispatch event to notify all components (including this one) to update
        globalThis.dispatchEvent(
          new CustomEvent("local-storage-changed", {
            detail: {
              key: keyRef.current,
              newValue: serializedValue,
              oldValue,
            },
          }),
        );
      } catch (error) {
        console.warn(
          `Failed to set localStorage key "${keyRef.current}":`,
          error,
        );
      }
    },
    [storedValue],
  );

  const removeValue = useCallback(() => {
    const oldValue = localStorage.getItem(keyRef.current);
    localStorage.removeItem(keyRef.current);

    globalThis.dispatchEvent(
      new CustomEvent("local-storage-changed", {
        detail: { key: keyRef.current, newValue: null, oldValue },
      }),
    );
  }, []);

  return [storedValue, setValue, removeValue];
}

function readFromStorage<T>(key: string, initialValue: T): T {
  const item = localStorage.getItem(key);
  if (item === null) return initialValue;
  if (item === "undefined") return undefined as unknown as T;
  return JSON.parse(item) as T;
}
