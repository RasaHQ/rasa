import {
  type CenterProps,
  type SpinnerProps,
  Center,
  Spinner,
  Text,
} from "@chakra-ui/react";

interface Props extends CenterProps {
  testId?: string;
  size?: SpinnerProps["size"];
  color?: SpinnerProps["color"];
  showText?: boolean;
}
export const LoadingSpinner = ({
  size = "xl",
  color,
  showText: showTextProp,
  ...props
}: Props) => {
  const showText = showTextProp ?? (size === "lg" || size === "xl");
  const defaultColor = "#574AE2";

  return (
    <Center height={"100%"} flexDirection="column" {...props}>
      <Spinner
        data-testid="loading-spinner"
        animationDuration="1s"
        color={color ?? defaultColor}
        size={size}
        mb={showText ? "1rem" : 0}
      />
      {showText ? <Text>Loading</Text> : null}
    </Center>
  );
};

