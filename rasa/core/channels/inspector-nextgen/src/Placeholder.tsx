import { Center, EmptyState, Image, VStack } from "@chakra-ui/react";
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
  children?: React.ReactNode;
}

export const NoData = ({
  shortLabel = "No results",
  longLabel = "We couldn't find any results to display.",
  image = PlaceholderImage.NoFlow,
  children,
}: Props) => {
  return (
    <Center height="100%">
      <EmptyState.Root>
        <EmptyState.Content>
          <Image src={images[image]} alt={shortLabel} height="56" mb="4" />
          <VStack gap="2" textAlign="center" maxW="21.25rem" mb="4">
            <EmptyState.Title>{shortLabel}</EmptyState.Title>
            {longLabel && (
              <EmptyState.Description>{longLabel}</EmptyState.Description>
            )}
          </VStack>
          {children}
        </EmptyState.Content>
      </EmptyState.Root>
    </Center>
  );
};
