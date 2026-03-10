"use client";

import {
  Toaster as ChakraToaster,
  Heading,
  Portal,
  Spinner,
  Stack,
  Toast,
} from "@chakra-ui/react";
import type { IconDefinition } from "@fortawesome/fontawesome-svg-core";
import { useCallback } from "react";
import { Icon } from "../Icon";
import { toaster } from "./toaster";

export const Toaster = () => {
  const renderIcon = useCallback((type: string, icon?: IconDefinition) => {
    if (icon) {
      return <Icon icon={icon} />;
    }
    if (type === "loading") {
      return <Spinner size="sm" color="blue.solid" />;
    }
    return <Toast.Indicator />;
  }, []);

  return (
    <Portal>
      <ChakraToaster
        toaster={toaster}
        insetInline={{ mdDown: "4" }}
        data-testid="toaster"
      >
        {(toast) => (
          <Toast.Root width={{ md: "sm" }}>
            {renderIcon(toast.type || "", toast.meta?.icon as IconDefinition)}
            <Stack gap="1" flex="1" maxWidth="100%">
              {toast.title && (
                <Toast.Title asChild>
                  <Heading color="inherit">{toast.title}</Heading>
                </Toast.Title>
              )}
              {toast.description && (
                <Toast.Description>{toast.description}</Toast.Description>
              )}
            </Stack>
            {toast.action && (
              <Toast.ActionTrigger>{toast.action.label}</Toast.ActionTrigger>
            )}
            {toast.closable && <Toast.CloseTrigger />}
          </Toast.Root>
        )}
      </ChakraToaster>
    </Portal>
  );
};
