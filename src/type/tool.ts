export type LinkItem = {
  name: string;
  link: string;
  avatar: string;
  descr: string;
};

export type TalkingItem = {
  date: string;
  tags: string[];
  content: string;
};

export type ToolDataSource<T> = {
  api: string;
  data: T[];
};

export type ToolSource<T> = string | T[];
