export const rasaShadows = {
  base: "0 0 0.25rem 0 rgba(0, 0, 0, 0.25)",
  focus: "0 0 0 0.25rem rgba(0, 0, 0, 0.25)",
  flowNode: "0 0.125rem 0.5rem rgba(0, 0, 0, 0.15)",
  flowNodeSelected: (color: string) => `inset 0 0 0 0.125rem ${color}`,
};
export type RasaShadows = typeof rasaShadows;
