import { Alert, Box } from "@chakra-ui/react";

export const ErrorAlert = ({ title, message }: { title: string; message?: string }) => {
  return (
    <Alert.Root status="error" title={title}>
      <Alert.Indicator />
      <Alert.Content>
        <Alert.Title>{title}</Alert.Title>
        {message &&
          <Alert.Description wordBreak="break-all">
            {message?.split("\\n")?.map(el => (
              <Box as="span" key="el">
                {el}
                <br />
              </Box>
            ))}
          </Alert.Description>
        }
      </Alert.Content>
    </Alert.Root >
  );
}
