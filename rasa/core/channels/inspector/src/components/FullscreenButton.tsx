import {
  IconButton,
  IconButtonProps,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalHeader,
  ModalOverlay,
  Tooltip,
  useDisclosure,
} from "@chakra-ui/react";
import { ExpandIcon } from "./ExpandIcon";

interface Props extends Omit<IconButtonProps, "aria-label"> {
  title: string;
}

export const FullscreenButton = ({ title, children, ...props }: Props) => {
  const { isOpen, onOpen, onClose } = useDisclosure();

  return (
    <>
      <Tooltip label="See in full screen" hasArrow>
        <IconButton
          size="sm"
          variant="outline"
          icon={<ExpandIcon />}
          {...props}
          aria-label="Full screen"
          onClick={onOpen}
        />
      </Tooltip>

      <Modal isOpen={isOpen} onClose={onClose} size="xl">
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>{title}</ModalHeader>
          <ModalCloseButton />
          <ModalBody>{children}</ModalBody>
        </ModalContent>
      </Modal>
    </>
  );
};
