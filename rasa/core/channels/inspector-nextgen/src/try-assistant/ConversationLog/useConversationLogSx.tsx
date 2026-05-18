export const useConversationLogSx = (isSelected: boolean) => {
  const containerSx = {
    flexDirection: "row",
    textAlign: "left",
    ml: 0,
    mr: 0,
    _first: { mt: 0 },
    p: "2",
    pl: "6",
    pr: "6",
    bg: isSelected ? "bg.muted" : "bg.panel",
  };

  const hoverableSx = {
    cursor: "pointer",
    _hover: {
      bg: "bg.muted",
      "& .message-bubble": {
        bg: "bg.panel",
      },
    },
  };

  const hoverableContainerSx = {
    ...containerSx,
    ...hoverableSx,
  };

  const iconSx = {
    marginRight: "6",
    height: "14",
    width: "14",
  };

  const baseMessageSx = {
    display: "flex",
    flexDirection: "row",
    alignItems: "center",
    borderRadius: "lg",
    color: "fg.muted",
  };

  return {
    containerSx,
    hoverableSx,
    hoverableContainerSx,
    baseMessageSx,
    iconSx,
  };
};
