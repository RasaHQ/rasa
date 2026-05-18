import { Box, ChakraProvider, Flex, Theme } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from 'react';
import { Background } from "./assets/images";
import { StandaloneHeader } from './components/StandaloneHeader';
import { Inspector } from './Inspector';
import { system } from './theme';
import { useSearchParams } from "react-router";

const App = () => {
  const [containerWidth, setContainerWidth] = useState("1000px");
  const queryClient = new QueryClient();
  const [searchParams] = useSearchParams();
  const containerCss = {
    background: `url(${Background})`,
    backgroundSize: "cover",
    height: "100vh",
    alignItems: "stretch",
  };

  const contentCss = {
    maxWidth: "100%",
    minWidth: "618px",
    width: containerWidth,
    transition: "width 0.4s ease",
    height: "calc(100% - 3rem)",
    margin: "0 auto",
  };

  // Inspector build is served from the Rasa server, so window.location.origin
  // already contains the correct host and specified --port 5007 e.g. http://localhost:5007.
  // Breaks in dev mode as origin is Vite dev server e.g. http://localhost:5173.
  // Therefore in dev mode default to hardcoded http://localhost:5005.
  const projectUrl = searchParams.get("projectUrl") || (import.meta.env.DEV ? "http://localhost:5005" : globalThis.location.origin);

  return (
    <QueryClientProvider client={queryClient}>
      <ChakraProvider value={system}>
        <Theme appearance="light">
          <Flex direction="column" css={containerCss}>
            <StandaloneHeader />
            <Box flex={1} css={contentCss} px="8" py="4" position="relative">
              <Inspector
                projectUrl={projectUrl}
                botDataEndpoint="/data"
                onInspectModeChange={(inspect) => setContainerWidth(inspect ? "100%" : "1000px")}
                singleSessionMode
              />
            </Box>
          </Flex>
        </Theme>
      </ChakraProvider>
    </QueryClientProvider >
  );
};

export default App;
