import { system } from "../theme";

const tokenToPx = (value: string): number => {
  if (value.endsWith("px")) return parseFloat(value);
  if (value.endsWith("rem")) return parseFloat(value) * 16;
  return parseFloat(value);
};

export const useTheme = () => {
  const getTokenPx = (token: string): number =>
    tokenToPx(system.token(token) as string);
  return { getToken: system.token, getTokenPx };
};
