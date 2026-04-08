import { Box, ChakraProvider, Flex } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Background } from "./assets/images";
import { StandaloneHeader } from './components/StandaloneHeader';
import { Inspector } from './Inspector';
import { system } from './theme';

const App = () => {
  const queryClient = new QueryClient();
  // Inspector build is served from the Rasa server, so window.location.origin
  // already contains the correct host and specified --port 5007 e.g. http://localhost:5007.
  // Breaks in dev mode as origin is Vite dev server e.g. http://localhost:5173.
  // Therefore in dev mode default to hardcoded http://localhost:5005.
  const projectUrl = import.meta.env.DEV ? "http://localhost:5005" : window.location.origin;
  return (
    <QueryClientProvider client={queryClient}>
      <ChakraProvider value={system}>
        <Flex
          direction="column"
          height="100vh"
          css={{
            background: `url(${Background})`,
            backgroundSize: "cover",
          }}
        >
          <StandaloneHeader />
          <Box flex={1} minHeight={0} px="2rem" py="1rem" position="relative">
            <Inspector
              projectUrl={projectUrl}
              botDataEndpoint="/data"
              singleSessionMode
            />
          </Box>
        </Flex>
      </ChakraProvider>
    </QueryClientProvider>
  );
};

export default App;
