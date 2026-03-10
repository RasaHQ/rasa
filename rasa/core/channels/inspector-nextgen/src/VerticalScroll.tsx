/* istanbul ignore file */
import { type BoxProps, Grid, GridItem, ScrollArea } from "@chakra-ui/react";

interface ScrollContainerProps extends BoxProps {
  children: React.ReactNode;
}
export const ScrollContainer = (props: ScrollContainerProps) => {
  const { children, css, ...otherProps } = props;
  const containerSx = {
    height: "100%",
    width: "100%",
    gridTemplateRows: `auto 1fr auto`,
    gridTemplateAreas: `"header" 
                        "content"
                        "footer"`,
    ...css,
  };

  return (
    <Grid css={containerSx} {...otherProps}>
      {children}
    </Grid>
  );
};

interface ScrollFixedHeaderProps extends BoxProps {
  children: React.ReactNode;
}
export const ScrollFixedHeader = (props: ScrollFixedHeaderProps) => {
  const { children, ...otherProps } = props;

  return (
    <GridItem area="header" {...otherProps}>
      {children}
    </GridItem>
  );
};

interface ScrollContentProps extends BoxProps {
  children: React.ReactNode;
  withSpacing?: boolean;
}
export const ScrollContent = (props: ScrollContentProps) => {
  const { children, withSpacing = true, css, ...otherProps } = props;

  const containerSx = {
    ...(withSpacing && {
      m: "1rem",
      mr: "0.5rem",
      pr: "1rem",
    }),
    ...css,
    overflow: "auto",
  };

  return (
    <GridItem area="content" css={containerSx} {...otherProps}>
      <ScrollArea.Root>
        <ScrollArea.Viewport>{children}</ScrollArea.Viewport>
        <ScrollArea.Scrollbar orientation="vertical">
          <ScrollArea.Thumb />
        </ScrollArea.Scrollbar>
      </ScrollArea.Root>
    </GridItem>
  );
};

interface ScrollFixedFooterProps extends BoxProps {
  children: React.ReactNode;
}
export const ScrollFixedFooter = (props: ScrollFixedFooterProps) => {
  const { children, ...otherProps } = props;

  return (
    <GridItem area="footer" {...otherProps}>
      {children}
    </GridItem>
  );
};

