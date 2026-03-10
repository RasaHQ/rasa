import { ChakraProvider, Flex } from '@chakra-ui/react';
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Background } from "./assets/images";
import { Inspector } from './Inspector';
import { system } from './theme';

const App = () => {
  const queryClient = new QueryClient();
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
              projectUrl="http://localhost:5005"
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
