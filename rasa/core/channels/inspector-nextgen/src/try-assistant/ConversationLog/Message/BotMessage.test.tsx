import { fireEvent, screen } from "@testing-library/react";
import { forwardRef } from "react";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/utils";
import { UtteranceType, type Utterance } from "../../../types";
import { BotMessage } from "./BotMessage";

vi.mock("../../../hooks/useTheme", () => ({
  useTheme: () => ({
    getToken: vi.fn().mockReturnValue("1rem"),
    getTokenPx: vi.fn().mockReturnValue(16),
  }),
}));

vi.mock("./MessageMarkup", () => ({
  MessageMarkup: forwardRef(
    (
      {
        onMouseEnter,
        onMouseLeave,
        topOverlay,
        children,
        messageBreakout,
      }: {
        onMouseEnter?: () => void;
        onMouseLeave?: () => void;
        topOverlay?: React.ReactNode;
        children?: React.ReactNode;
        messageBreakout?: React.ReactNode;
        [key: string]: unknown;
      },
      ref: React.Ref<HTMLDivElement>,
    ) => (
      <div
        data-testid="message-markup"
        ref={ref}
        onMouseEnter={onMouseEnter}
        onMouseLeave={onMouseLeave}
      >
        {topOverlay}
        {children}
        {messageBreakout}
      </div>
    ),
  ),
}));

vi.mock("../../../Modal", () => ({
  Modal: ({
    button,
    children,
  }: {
    button?: () => React.ReactNode;
    children: React.ReactNode;
    [key: string]: unknown;
  }) => (
    <>
      {button?.()}
      {children}
    </>
  ),
}));

vi.mock("../../../RasaCodeBlock", () => ({
  RasaCodeBlock: ({ code }: { code: string }) => (
    <pre data-testid="code-block">{code}</pre>
  ),
}));

vi.mock("../../../ButtonLink", () => ({
  ButtonLink: ({
    children,
    to,
    ...props
  }: {
    children: React.ReactNode;
    to: string;
    [key: string]: unknown;
  }) => (
    <a href={to} {...(props as React.AnchorHTMLAttributes<HTMLAnchorElement>)}>
      {children}
    </a>
  ),
}));

const baseUtterance: Omit<Utterance, "entities"> = {
  __typename: "Utterance",
  id: "u1",
  type: UtteranceType.Bot,
  text: "Hello there!",
  timestamp: new Date().toISOString(),
  originalTimestamp: 0,
  tokens: [],
  rephrase: false,
  rephrasePrompt: null,
  metadata: { parseData: {} },
  responseData: { buttons: [], quickReplies: [] },
};

describe("BotMessage", () => {
  describe("text rendering", () => {
    it("renders the utterance text", () => {
      renderWithProviders(
        <BotMessage utterance={baseUtterance} inspectorMode={false} />,
      );
      expect(screen.getByText("Hello there!")).toBeInTheDocument();
    });

    it("does not render a text node when text is empty", () => {
      const utterance = { ...baseUtterance, text: "" };
      renderWithProviders(
        <BotMessage utterance={utterance} inspectorMode={false} />,
      );
      expect(screen.queryByText("Hello there!")).not.toBeInTheDocument();
    });
  });

  describe("image rendering", () => {
    it("renders an image when responseData.image is provided", () => {
      const utterance = {
        ...baseUtterance,
        responseData: {
          ...baseUtterance.responseData!,
          image: "https://example.com/photo.png",
        },
      };
      renderWithProviders(
        <BotMessage utterance={utterance} inspectorMode={false} />,
      );
      expect(screen.getByRole("img", { name: /assistant image/i })).toBeInTheDocument();
    });

    it("does not render an image when responseData.image is absent", () => {
      renderWithProviders(
        <BotMessage utterance={baseUtterance} inspectorMode={false} />,
      );
      expect(screen.queryByRole("img")).not.toBeInTheDocument();
    });
  });

  describe("buttons and quick replies", () => {
    it("renders a button from responseData.buttons", () => {
      const utterance = {
        ...baseUtterance,
        responseData: {
          buttons: [{ title: "Yes please", payload: "/affirm" }],
          quickReplies: [],
        },
      };
      renderWithProviders(
        <BotMessage utterance={utterance} inspectorMode={false} />,
      );
      expect(screen.getByRole("button", { name: "Yes please" })).toBeInTheDocument();
    });

    it("renders a quick reply button", () => {
      const utterance = {
        ...baseUtterance,
        responseData: {
          buttons: [],
          quickReplies: [{ title: "Maybe", payload: "/maybe" }],
        },
      };
      renderWithProviders(
        <BotMessage utterance={utterance} inspectorMode={false} />,
      );
      expect(screen.getByRole("button", { name: "Maybe" })).toBeInTheDocument();
    });

    it("calls onQuickReply with the button payload when clicked", () => {
      const onQuickReply = vi.fn();
      const utterance = {
        ...baseUtterance,
        responseData: {
          buttons: [{ title: "Yes", payload: "/affirm" }],
          quickReplies: [],
        },
      };
      renderWithProviders(
        <BotMessage
          utterance={utterance}
          onQuickReply={onQuickReply}
          isInteractive
          inspectorMode={false}
        />,
      );
      fireEvent.click(screen.getByRole("button", { name: "Yes" }));
      expect(onQuickReply).toHaveBeenCalledWith("/affirm");
    });

    it("disables buttons when isInteractive is false", () => {
      const utterance = {
        ...baseUtterance,
        responseData: {
          buttons: [{ title: "Yes", payload: "/affirm" }],
          quickReplies: [],
        },
      };
      renderWithProviders(
        <BotMessage
          utterance={utterance}
          isInteractive={false}
          inspectorMode={false}
        />,
      );
      expect(screen.getByRole("button", { name: "Yes" })).toBeDisabled();
    });

    it("renders a link for buttons whose payload starts with a URL protocol", () => {
      const utterance = {
        ...baseUtterance,
        responseData: {
          buttons: [{ title: "Visit site", payload: "https://example.com" }],
          quickReplies: [],
        },
      };
      renderWithProviders(
        <BotMessage utterance={utterance} inspectorMode={false} />,
      );
      expect(screen.getByRole("link", { name: /Visit site/ })).toBeInTheDocument();
    });

    it("renders a link for attachment in responseData", () => {
      const utterance = {
        ...baseUtterance,
        responseData: {
          buttons: [],
          quickReplies: [],
          attachment: "report.pdf",
        },
      };
      renderWithProviders(
        <BotMessage utterance={utterance} inspectorMode={false} />,
      );
      expect(screen.getByRole("link", { name: /report\.pdf/ })).toBeInTheDocument();
    });
  });

  describe("custom JSON payload", () => {
    const utteranceWithCustom = {
      ...baseUtterance,
      text: "",
      responseData: {
        buttons: [],
        quickReplies: [],
        custom: { key: "value" },
      },
    };

    it("shows the JSON label when custom is provided", () => {
      renderWithProviders(
        <BotMessage utterance={utteranceWithCustom} inspectorMode={false} />,
      );
      expect(screen.getByText("JSON")).toBeInTheDocument();
    });

    it("shows the 'Click to view code' button when custom is provided", () => {
      renderWithProviders(
        <BotMessage utterance={utteranceWithCustom} inspectorMode={false} />,
      );
      expect(screen.getByRole("button", { name: /click to view code/i })).toBeInTheDocument();
    });

    it("renders a separator when both text and custom are present", () => {
      const utterance = {
        ...baseUtterance,
        text: "Some text",
        responseData: {
          buttons: [],
          quickReplies: [],
          custom: { key: "value" },
        },
      };
      renderWithProviders(
        <BotMessage utterance={utterance} inspectorMode={false} />,
      );
      expect(screen.getByRole("separator")).toBeInTheDocument();
    });

    it("does not render a separator when only text is present", () => {
      renderWithProviders(
        <BotMessage utterance={baseUtterance} inspectorMode={false} />,
      );
      expect(screen.queryByRole("separator")).not.toBeInTheDocument();
    });

    it("does not render a separator when only custom is present", () => {
      renderWithProviders(
        <BotMessage utterance={utteranceWithCustom} inspectorMode={false} />,
      );
      expect(screen.queryByRole("separator")).not.toBeInTheDocument();
    });
  });

  describe("hover action button", () => {
    const actions = [{ icon: {} as never, label: "Copy", action: vi.fn() }];

    it("does not show action button before hovering", () => {
      renderWithProviders(
        <BotMessage
          utterance={baseUtterance}
          inspectorMode={false}
          conversationEventActions={actions}
        />,
      );
      expect(screen.queryByRole("button", { name: "Copy" })).not.toBeInTheDocument();
    });

    it("shows action button on mouse enter", () => {
      renderWithProviders(
        <BotMessage
          utterance={baseUtterance}
          inspectorMode={false}
          conversationEventActions={actions}
        />,
      );
      fireEvent.mouseEnter(screen.getByTestId("message-markup"));
      expect(screen.getByRole("button", { name: "Copy" })).toBeInTheDocument();
    });

    it("hides action button after mouse leave", () => {
      renderWithProviders(
        <BotMessage
          utterance={baseUtterance}
          inspectorMode={false}
          conversationEventActions={actions}
        />,
      );
      const markup = screen.getByTestId("message-markup");
      fireEvent.mouseEnter(markup);
      fireEvent.mouseLeave(markup);
      expect(screen.queryByRole("button", { name: "Copy" })).not.toBeInTheDocument();
    });

    it("does not show action button when conversationEventActions is empty", () => {
      renderWithProviders(
        <BotMessage
          utterance={baseUtterance}
          inspectorMode={false}
          conversationEventActions={[]}
        />,
      );
      fireEvent.mouseEnter(screen.getByTestId("message-markup"));
      expect(screen.queryByRole("button", { name: "Copy" })).not.toBeInTheDocument();
    });
  });
});
