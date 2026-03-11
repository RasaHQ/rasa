import { useContext } from "react";
import { InspectorContext, type InspectorContextValue } from "./InspectorContext";

export const useInspectorContext = (): InspectorContextValue =>
  useContext(InspectorContext);
