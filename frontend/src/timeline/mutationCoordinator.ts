export type VersionedTimelineResponse = {
  project_id: string;
  version: number;
};

export type TimelineMutationCoordinator = {
  activate(projectId: string | null): void;
  run<T extends VersionedTimelineResponse>(
    projectId: string,
    getCurrentVersion: () => number,
    request: (expectedVersion: number) => Promise<T>,
    apply: (response: T) => void,
  ): Promise<T>;
};

export class TimelineMutationProjectChangedError extends Error {
  constructor() {
    super("Timeline mutation project changed before response completed");
    this.name = "TimelineMutationProjectChangedError";
  }
}

export function createTimelineMutationCoordinator(): TimelineMutationCoordinator {
  let activeProjectId: string | null = null;
  let generation = 0;
  let tail: Promise<void> = Promise.resolve();
  const activate = (projectId: string | null): void => {
    if (activeProjectId === projectId) return;
    activeProjectId = projectId;
    generation += 1;
    tail = Promise.resolve();
  };

  return {
    activate,

    run<T extends VersionedTimelineResponse>(
      projectId: string,
      getCurrentVersion: () => number,
      request: (expectedVersion: number) => Promise<T>,
      apply: (response: T) => void,
    ): Promise<T> {
      if (activeProjectId !== projectId) {
        return Promise.reject(new TimelineMutationProjectChangedError());
      }
      const requestGeneration = generation;
      const task = tail.then(async () => {
        const response = await request(getCurrentVersion());
        if (
          generation !== requestGeneration ||
          activeProjectId !== projectId ||
          response.project_id !== projectId
        ) {
          throw new TimelineMutationProjectChangedError();
        }
        if (response.version >= getCurrentVersion()) {
          apply(response);
        }
        return response;
      });
      tail = task.then(
        () => undefined,
        () => undefined,
      );
      return task;
    },
  };
}
