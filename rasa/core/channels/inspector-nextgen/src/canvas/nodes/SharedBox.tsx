import { type FlexProps, Flex } from "@chakra-ui/react";

export const SharedBox = (props: FlexProps) => {
  const sx = {
    bg: "#FFFFFF",
    color: "#2C3951",
    borderRadius: "1rem",
    overflow: "hidden",
    height: "100%",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "column",
    border: "1px solid",
    "@supports (font: -apple-system-body) and (-webkit-appearance: none)": {
      // safari has issues rendering the shadow in an svg.
      // Since it's just cosmetics, we can remove it.
      boxShadow: "none",
    },
    ...props.css,
  };

  return (
    <Flex {...props} css={sx}>
      {props.children}
    </Flex>
  );
};
