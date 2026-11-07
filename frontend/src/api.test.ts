import { describe, expect, it } from "vitest";
import { SseParser } from "./api";

const frame = 'event: token\ndata: {"text": "Hi"}\n\n';

describe("SseParser", () => {
  it("reads one event", () => {
    expect(new SseParser().feed(frame)).toEqual([{ event: "token", data: { text: "Hi" } }]);
  });

  it("reads several events from one chunk", () => {
    expect(new SseParser().feed(frame + frame)).toHaveLength(2);
  });

  it("waits for the rest of an event split across chunks", () => {
    for (let cut = 1; cut < frame.length; cut++) {
      const parser = new SseParser();
      const events = [...parser.feed(frame.slice(0, cut)), ...parser.feed(frame.slice(cut))];
      expect(events).toEqual([{ event: "token", data: { text: "Hi" } }]);
    }
  });

  it("holds an unfinished event back", () => {
    expect(new SseParser().feed('event: token\ndata: {"text"')).toEqual([]);
  });

  it("keeps newlines that are inside the JSON text", () => {
    const [event] = new SseParser().feed('event: token\ndata: {"text": "a\\n\\nb"}\n\n');
    expect(event.data.text).toBe("a\n\nb");
  });
});
