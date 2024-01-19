import type { ComponentStyleConfig } from "@chakra-ui/theme";
import { mode } from "@chakra-ui/theme-tools";

export const Modal: ComponentStyleConfig = {
  baseStyle: (props) => {
    const { theme } = props;

    return {
      dialog: {
        borderRadius: theme.rasaRadii.large,
        fontSize: theme.rasaFontSizes.md,
        width: "75%",
        height: "85%",
        margin: "auto",
      },
      dialogContainer: {
        zIndex: theme.zIndices.modal,
        "> section": {
          maxW: "100%",
        },
      },
      overlay: {
        zIndex: theme.zIndices.modalOverlay,
      },
      header: {
        borderBottom: "1px solid",
        borderColor: mode("neutral.300", "neutral.300")(props),
        px: theme.rasaSpace[2],
        pt: theme.rasaSpace[2],
        pb: theme.rasaSpace[1.5],
        // make sure the header does not overlap with the close button
        pr: theme.rasaSpace[3],
        lineHeight: 1,
        fontSize: theme.rasaFontSizes.lg,
      },
      body: {
        px: theme.rasaSpace[2],
        py: theme.rasaSpace[1.5],
        overflow: "auto",
      },
      closeButton: {
        right: theme.rasaSpace[1],
        top: theme.rasaSpace[1],
      },
    };
  },
};
