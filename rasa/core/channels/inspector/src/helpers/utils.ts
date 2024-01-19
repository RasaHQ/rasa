import { SelectedStack, Stack } from "../types";

export const shouldShowTooltip = (text: string) => {
  const textLength = text.length;

  if (textLength > 10 && textLength < 89) {
    return true;
  }

  return false;
};

export const updatedActiveFrame = (previous: SelectedStack | undefined, updatedStack: Stack[]) => {
  // try to find the currently active frame in the updated stack
  // if it isn't there anymore, we will show the first non-pattern frame
  // instead

  // reset previously active stack frame, if it was not user selected
  if(!previous?.isUserSelected){
    previous = undefined
  }

  const activeFrame = updatedStack.find(
    (stackFrame) => stackFrame.frame_id === previous?.stack.frame_id
  );
  if (!activeFrame) {
    const updatedFrame = updatedStack
      .slice()
      .reverse()
      .find((frame) => !frame.flow_id?.startsWith("pattern_"));
    if(updatedFrame !== undefined) {
      return {
        stack: updatedFrame,
        isUserSelected: false,
      };
    } else {
      return undefined;
    }
  } else {
    return previous;
  }
};
