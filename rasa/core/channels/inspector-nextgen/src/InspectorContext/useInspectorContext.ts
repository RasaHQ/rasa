import { useContext } from "react";
import { InspectorContext } from "./InspectorContext";

export const useInspectorContext = () => useContext(InspectorContext);
