import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

// The panes themselves are tested on their own; here only the layout's choice matters.
vi.mock("./panes/ChatPane", () => ({ ChatPane: () => <div /> }));
vi.mock("./panes/LibraryPane", () => ({ LibraryPane: () => <div /> }));
vi.mock("./panes/ViewerPane", () => ({
  ViewerPane: () => <div data-testid="viewer" />,
  ViewerStrip: () => <div data-testid="strip" />,
}));

const { App } = await import("./App");
const { useStore } = await import("./store/store");

afterEach(cleanup);

describe("the viewer column (item 9)", () => {
  it("is a narrow strip while nothing is open, and full width once something is", () => {
    useStore.setState({ viewer: null, shownExplanation: null, loadDocuments: async () => {}, loadOptions: async () => {} });
    const { container } = render(<App />);
    expect(screen.getByTestId("strip")).toBeTruthy();
    expect((container.firstChild as HTMLElement).className).toContain("_2.5rem]");
    act(() => useStore.getState().openPage(3, 7));
    expect(screen.getByTestId("viewer")).toBeTruthy();
    expect((container.firstChild as HTMLElement).className).toContain("minmax(380px,1fr)_minmax(380px,1fr)");
    act(() => useStore.getState().closeViewer());
    expect(screen.getByTestId("strip")).toBeTruthy();
  });

  it("opens for an explanation even with no document open", () => {
    useStore.setState({ viewer: null, shownExplanation: 5, loadDocuments: async () => {}, loadOptions: async () => {} });
    render(<App />);
    expect(screen.getByTestId("viewer")).toBeTruthy();
  });
});
