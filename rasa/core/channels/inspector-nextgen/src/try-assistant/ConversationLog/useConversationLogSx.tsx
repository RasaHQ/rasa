export const useConversationLogSx = (isSelected: boolean) => {
  const containerSxBgSelected = "#F6F7FB";
  const containerSxBg = "#FFFFFF";

  const hoverBg = "rasaNeutral.100";

  const containerSx = {
    flexDirection: "row",
    textAlign: "left",
    ml: 0,
    mr: 0,
    _first: { mt: 0 },
    p: "0.5rem",
    pl: "1.5rem",
    pr: "1.5rem",
    bg: isSelected ? containerSxBgSelected : containerSxBg,
  };

  const hoverableSx = {
    cursor: "pointer",
    _hover: {
      bg: hoverBg,
      "& .message-bubble": {
        bg: "rasaNeutral.50",
      },
    },
  };

  const hoverableContainerSx = {
    ...containerSx,
    ...hoverableSx,
  };

  const iconSx = {
    marginRight: "0.5rem",
  };

  const baseMessageSx = {
    display: "flex",
    flexDirection: "row",
    alignItems: "center",
    borderRadius: "0.5rem",
  };

  return {
    containerSx,
    hoverableSx,
    hoverableContainerSx,
    baseMessageSx,
    iconSx,
  };
};
