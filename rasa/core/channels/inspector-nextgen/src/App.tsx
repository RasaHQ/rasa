import { ChakraProvider, Flex } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Background } from "./assets/images";
import { Inspector } from './Inspector';
import { system } from './theme';

const App = () => {
  const queryClient = new QueryClient();
  // Inspector build is served from the Rasa server, so window.location.origin
  // already contains the correct host and specified --port 5007 e.g. http://localhost:5007.
  // Breaks in dev mode as origin is Vite dev server e.g. http://localhost:5173.
  // Therefore in dev mode default to hardcoded http://localhost:5005.
  const projectUrl = import.meta.env.DEV ? "http://localhost:5005" : window.location.origin;
  const containerCss = {
    background: `url(${Background})`,
    backgroundSize: "cover",
    height: "100vh",
    alignItems: "center",
  };

  const contentCss = {
    width: "66%",
    height: "calc(100% - 4rem)",
    minWidth: "618px",
    margin: "0 auto",
    borderColor: "gray.200",
    bg: "rasaNeutral.50",
    borderRadius: "1rem",
    justifyContent: "center",
  };

  return (
    <QueryClientProvider client={queryClient}>
      <ChakraProvider value={system}>
        <Flex css={containerCss}>
          <Flex css={contentCss}>
            <Inspector
              projectUrl={projectUrl}
              botDataEndpoint="/data"
              singleSessionMode
            />
          </Flex>
        </Flex>
      </ChakraProvider>
    </QueryClientProvider>
  );
};

export default App;
