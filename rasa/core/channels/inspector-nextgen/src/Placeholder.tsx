import { Center, Text, Box, Heading, Image } from "@chakra-ui/react";
import {
  NoFlow,
  Mobile,
  Cubes,
  CubesError,
  CubesPause,
} from "./assets/images";
import { PlaceholderImage } from "./types";

const images = {
  [PlaceholderImage.NoFlow]: NoFlow,
  [PlaceholderImage.Mobile]: Mobile,
  [PlaceholderImage.Cubes]: Cubes,
  [PlaceholderImage.CubesError]: CubesError,
  [PlaceholderImage.CubesPause]: CubesPause,
};

interface Props {
  shortLabel?: string;
  longLabel?: string;
  image?: PlaceholderImage;
  title?: string;
  children?: React.ReactNode;
}

export const NoData = ({
  shortLabel = "No results",
  longLabel = "We couldn't find any results to display.",
  image = PlaceholderImage.NoFlow,
  children,
  ...props
}: Props) => {
  const boxSx = {
    display: "grid",
    rowGap: "0.5rem",
    maxWidth: "21.25rem",
    mb: "1rem",
  };
  const textColor = "rasaNeutral.700";

  return (
    <Center height={"100%"} flexDirection="column" {...props}>
      <Image src={images[image]} alt={shortLabel} height="14rem" mb={6} />
      <Box css={boxSx}>
        <Heading fontSize="1rem" textAlign="center">
          {shortLabel}
        </Heading>
        {longLabel && (
          <Text color={textColor} textAlign="center">
            {longLabel}
          </Text>
        )}
      </Box>
      {children}
    </Center>
  );
};

