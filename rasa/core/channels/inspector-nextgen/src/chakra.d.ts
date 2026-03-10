import type { RecipeVariantProps } from "@chakra-ui/react";

declare module "@chakra-ui/react" {
  interface TextProps extends RecipeVariantProps<"text"> {
    variant?: "primary" | "muted";
    size?: "xs" | "sm" | "md" | "xxl";
  }

  interface HeadingProps extends RecipeVariantProps<"heading"> {
    size?: "sm" | "md" | "lg" | "xxl";
  }
}

