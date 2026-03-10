import {
  Avatar as ChakraAvatar,
  AvatarGroup as ChakraAvatarGroup,
} from "@chakra-ui/react";
import * as React from "react";

type ImageProps = React.ImgHTMLAttributes<HTMLImageElement>;

export interface AvatarProps extends ChakraAvatar.RootProps {
  name?: string;
  src?: string;
  srcSet?: string;
  loading?: ImageProps["loading"];
  icon?: React.ReactElement;
  fallback?: React.ReactNode;
  objectFit?: "cover" | "contain" | "fill" | "none" | "scale-down";
}

export const Avatar = React.forwardRef<HTMLDivElement, AvatarProps>(
  function Avatar(props, ref) {
    const {
      name,
      src,
      srcSet,
      loading,
      icon,
      fallback,
      children,
      objectFit,
      ...rest
    } = props;
    return (
      <ChakraAvatar.Root ref={ref} {...rest}>
        <ChakraAvatar.Fallback name={name}>
          {fallback || icon}
        </ChakraAvatar.Fallback>
        <ChakraAvatar.Image
          src={src}
          srcSet={srcSet}
          objectFit={objectFit}
          loading={loading}
        />
        {children}
      </ChakraAvatar.Root>
    );
  },
);

export const AvatarGroup = ChakraAvatarGroup;

