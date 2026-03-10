import {
  type ButtonProps,
  type DialogRootProps,
  Box,
  Button,
  Dialog,
  Flex,
  Heading,
  HStack,
  IconButton,
  Portal,
} from "@chakra-ui/react";
import { ButtonLink } from "./ButtonLink";
import { Icon, XMark } from "./Icon";
import { LoadingSpinner } from "./LoadingSpinner";
import { Tooltip } from "./Tooltip";

interface Action extends ButtonProps {
  title: string;
  id: string;
  type?: "button" | "submit" | "reset";
  tooltip?: string;
  onClick?: (e: React.MouseEvent<HTMLElement>) => void;
  linksTo?: string;
}

interface Props extends DialogRootProps {
  button?: () => React.ReactNode | string;
  title?: string;
  children: React.ReactNode;
  onClose: () => void;
  onOpen: () => void;
  actions?: Action[];
  type?: "form" | "div";
  displayOverlay?: boolean;
  footer?: React.ReactNode;
  isLoading?: boolean;
  canBeClosed?: boolean;
  tooltipContent?: React.ReactNode;
}

export const Modal = (props: Props) => {
  const {
    open,
    button,
    onClose,
    onOpen,
    title,
    children,
    actions,
    type = "div",
    displayOverlay = true,
    footer,
    isLoading = false,
    canBeClosed = true,
    tooltipContent,
    ...otherProps
  } = props;

  const hasActions = actions?.length;

  // if cant be closed set the closeOnEscape and closeOnInteractOutside to false
  if (!canBeClosed) {
    otherProps.closeOnEscape = false;
    otherProps.closeOnInteractOutside = false;
  }

  const overflowSx = {
    display: "flex",
    flexDirection: "column",
    maxHeight: "100%",
    overflow: "hidden",
    flexGrow: 1,
  };

  return (
    <Dialog.Root
      placement="center"
      open={open}
      onOpenChange={() => (open ? onClose() : onOpen())}
      {...otherProps}
    >
      {button &&
        (tooltipContent ? (
          <Tooltip content={tooltipContent} showArrow>
            <Dialog.Trigger asChild>{button()}</Dialog.Trigger>
          </Tooltip>
        ) : (
          <Dialog.Trigger asChild>{button()}</Dialog.Trigger>
        ))}
      <Portal>
        {displayOverlay && <Dialog.Backdrop />}
        <Dialog.Positioner>
          <Dialog.Content>
            <Dialog.Header pt="1rem">
              <Heading size="lg">{title}</Heading>
            </Dialog.Header>
            {isLoading ? (
              <Flex flexGrow={1} justifyContent="center" alignItems="center">
                <LoadingSpinner />
              </Flex>
            ) : (
              <Box as={type} css={overflowSx}>
                <Dialog.Body p="2rem" pt="0.5rem">
                  {children}
                </Dialog.Body>
                {hasActions || footer ? (
                  <Dialog.Footer>
                    {footer}
                    {hasActions ? <Actions actions={actions} /> : null}
                  </Dialog.Footer>
                ) : null}
                {canBeClosed && (
                  <Dialog.CloseTrigger asChild>
                    <IconButton
                      size="lg"
                      fontSize="lg"
                      variant="subtle"
                      colorPalette="dark"
                      aria-label="Close"
                    >
                      <Icon icon={XMark} />
                    </IconButton>
                  </Dialog.CloseTrigger>
                )}
              </Box>
            )}
          </Dialog.Content>
        </Dialog.Positioner>
      </Portal>
    </Dialog.Root>
  );
};

const Actions = ({ actions }: { actions: Action[] }) => {
  return (
    <HStack gap="0.5rem" flexGrow={1} justifyContent="flex-end">
      {actions.map((action) => {
        const {
          id,
          title,
          onClick,
          type = "button",
          linksTo,
          ...props
        } = action;
        const handleClick = (e: React.MouseEvent<HTMLElement>) => {
          e.preventDefault();
          onClick?.(e);
        };

        if (linksTo) {
          return (
            <ButtonLink key={id} to={linksTo} variant={props.variant}>
              {title}
            </ButtonLink>
          );
        }

        return (
          <Button
            key={id}
            type={type}
            onClick={handleClick}
            borderRadius={"0.5rem"}
            {...props}
          >
            {title}
          </Button>
        );
      })}
    </HStack>
  );
};

