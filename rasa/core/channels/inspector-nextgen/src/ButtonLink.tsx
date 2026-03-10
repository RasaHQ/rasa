import React from "react";
import {
  type LinkProps as ReactLinkProps,
  Link as RouterLink,
} from "react-router-dom";
import {
  type ButtonProps as ChakraButtonProps,
  Button,
} from "@chakra-ui/react";

type ButtonProps = ReactLinkProps & ChakraButtonProps;

export const ButtonLink = ({ onClick, ...props }: ButtonProps) => {
  const clickEventHandler = (event: React.MouseEvent<HTMLButtonElement>) => {
    if (props.disabled) {
      event.preventDefault();
    } else {
      onClick?.(event);
    }
  };

  return (
    <Button
      as={RouterLink}
      tabIndex={0}
      {...props}
      onClick={clickEventHandler}
    />
  );
};

