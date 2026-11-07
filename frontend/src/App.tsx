import { useEffect } from "react";
import { ChatPane } from "./panes/ChatPane";
import { LibraryPane } from "./panes/LibraryPane";
import { ViewerPane, ViewerStrip } from "./panes/ViewerPane";
import { useStore } from "./store/store";

const LIBRARY_REFRESH_MS = 4000;

export function App() {
  const loadDocuments = useStore((state) => state.loadDocuments);
  const loadOptions = useStore((state) => state.loadOptions);
  useEffect(() => {
    void loadDocuments();
    void loadOptions();
    // The worker runs beside the server, so the library changes by itself: new files,
    // stages finishing, an OK awaited. A local request every few seconds keeps up with it.
    const timer = setInterval(() => void loadDocuments(), LIBRARY_REFRESH_MS);
    return () => clearInterval(timer);
  }, [loadDocuments, loadOptions]);

  // With nothing open the viewer is a narrow strip, and the chat takes the room (item 9).
  const collapsed = useStore((state) => state.viewer === null && state.shownExplanation === null);
  return (
    <div className={`grid h-dvh min-w-[900px] overflow-hidden ${collapsed
      ? "grid-cols-[250px_minmax(380px,1fr)_2.5rem]"
      : "grid-cols-[250px_minmax(380px,1fr)_minmax(380px,1fr)]"}`}>
      <LibraryPane />
      <ChatPane />
      {collapsed ? <ViewerStrip /> : <ViewerPane />}
    </div>
  );
}
