import { describe, expect, it } from "vitest";

import { createTimelineMutationCoordinator } from "./mutationCoordinator";


describe("timeline mutation coordinator", () => {
  it("serializes requests and gives the next request the applied version", async () => {
    let currentVersion = 2;
    let resolveFirst:
      | ((response: {
          project_id: string;
          version: number;
          value: string;
        }) => void)
      | undefined;
    const calls: number[] = [];
    const applied: string[] = [];
    const coordinator = createTimelineMutationCoordinator();
    coordinator.activate("project-a");

    const first = coordinator.run(
      "project-a",
      () => currentVersion,
      (expectedVersion) => {
        calls.push(expectedVersion);
        return new Promise<{
          project_id: string;
          version: number;
          value: string;
        }>((resolve) => {
          resolveFirst = resolve;
        });
      },
      (response) => {
        currentVersion = response.version;
        applied.push(response.value);
      },
    );
    const second = coordinator.run(
      "project-a",
      () => currentVersion,
      async (expectedVersion) => {
        calls.push(expectedVersion);
        return { project_id: "project-a", version: 4, value: "second" };
      },
      (response) => {
        currentVersion = response.version;
        applied.push(response.value);
      },
    );

    await Promise.resolve();
    expect(calls).toEqual([2]);
    resolveFirst?.({
      project_id: "project-a",
      version: 3,
      value: "first",
    });
    await Promise.all([first, second]);

    expect(calls).toEqual([2, 3]);
    expect(applied).toEqual(["first", "second"]);
    expect(currentVersion).toBe(4);
  });

  it("discards a response older than the currently applied version", async () => {
    let currentVersion = 5;
    const applied: number[] = [];
    const coordinator = createTimelineMutationCoordinator();
    coordinator.activate("project-a");

    await coordinator.run(
      "project-a",
      () => currentVersion,
      async () => ({ project_id: "project-a", version: 4 }),
      (response) => applied.push(response.version),
    );

    expect(applied).toEqual([]);
    expect(currentVersion).toBe(5);
  });

  it("drops delayed old-project work after activating a new project", async () => {
    let currentVersion = 1;
    let resolveProjectA:
      | ((response: { project_id: string; version: number }) => void)
      | undefined;
    const applied: string[] = [];
    const expectedVersions: Array<[string, number]> = [];
    const coordinator = createTimelineMutationCoordinator();
    coordinator.activate("project-a");

    const projectA = coordinator.run(
      "project-a",
      () => currentVersion,
      (expectedVersion) => {
        expectedVersions.push(["project-a", expectedVersion]);
        return new Promise<{ project_id: string; version: number }>(
          (resolve) => {
            resolveProjectA = resolve;
          },
        );
      },
      (response) => {
        currentVersion = response.version;
        applied.push(response.project_id);
      },
    );
    await Promise.resolve();

    currentVersion = 7;
    coordinator.activate("project-b");
    const projectB = coordinator.run(
      "project-b",
      () => currentVersion,
      async (expectedVersion) => {
        expectedVersions.push(["project-b", expectedVersion]);
        return { project_id: "project-b", version: 8 };
      },
      (response) => {
        currentVersion = response.version;
        applied.push(response.project_id);
      },
    );
    await projectB;
    resolveProjectA?.({ project_id: "project-a", version: 99 });
    await expect(projectA).rejects.toThrow("project changed");

    expect(expectedVersions).toEqual([
      ["project-a", 1],
      ["project-b", 7],
    ]);
    expect(applied).toEqual(["project-b"]);
    expect(currentVersion).toBe(8);
  });

  it("rejects a response whose project id differs from the request", async () => {
    let currentVersion = 3;
    const applied: string[] = [];
    const coordinator = createTimelineMutationCoordinator();
    coordinator.activate("project-a");

    await expect(
      coordinator.run(
        "project-a",
        () => currentVersion,
        async () => ({ project_id: "project-b", version: 4 }),
        (response) => {
          currentVersion = response.version;
          applied.push(response.project_id);
        },
      ),
    ).rejects.toThrow("project changed");

    expect(applied).toEqual([]);
    expect(currentVersion).toBe(3);
  });
});
