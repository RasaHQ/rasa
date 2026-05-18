import type { RecipeVariantProps } from "@chakra-ui/react";

declare module "@chakra-ui/react" {
  interface TextProps extends RecipeVariantProps<"text"> {
    variant?: "primary" | "muted";
    size?: "xs" | "sm" | "md" | "lg" | "xl" | "2xl" | "xxl";
  }

  interface HeadingProps extends RecipeVariantProps<"heading"> {
    size?: "xs" | "sm" | "md" | "lg" | "xl" | "2xl" | "xxl";
  }

  interface InputProps {
    size?: "2xs" | "xs" | "sm" | "md" | "lg" | "xl" | "2xl" | "3xl";
  }
}
