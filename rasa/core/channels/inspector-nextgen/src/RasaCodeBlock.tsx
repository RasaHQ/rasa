import { CodeBlock as ChakraCodeBlock, type CodeBlockRootProps } from "@chakra-ui/react";

type CodeBlockProps = Omit<CodeBlockRootProps, "children"> & {
  code: string;
}

export const RasaCodeBlock = ({
  code,
  ...props
}: CodeBlockProps) => {
  return (
    <ChakraCodeBlock.Root
      code={code}
      fontFamily="mono"
      size="sm"
      {...props}
    >
      <ChakraCodeBlock.Code>
        <ChakraCodeBlock.CodeText />
      </ChakraCodeBlock.Code>
    </ChakraCodeBlock.Root>
  );
}
