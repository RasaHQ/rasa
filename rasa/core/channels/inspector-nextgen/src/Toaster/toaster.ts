import { createToaster } from "@chakra-ui/react";
import type { ShowToastFn } from "../types";

export const toaster = createToaster({
  placement: "bottom-end",
  pauseOnPageIdle: true,
});

export const defaultShowToast: ShowToastFn = (options) => {
  toaster.create(options);
};
