type TopableEntry = {
  data: {
    top?: boolean;
  };
};

export default <T extends TopableEntry>(arr: T[]) => {
  const index = arr.findIndex((item) => item.data.top === true);
  if (index !== -1) {
    const [item] = arr.splice(index, 1);
    arr.unshift(item);
  }
  return arr;
}
