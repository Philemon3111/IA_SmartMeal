import numpy as np

def make_packet(df, valid_idx, scores, N):

    if scores is None:
        scores = np.ones(len(valid_idx)) / len(valid_idx)  # Uniform scores if None
    # Pair (row, score) where row is a recipe from the DataFrame
    items = sorted(
        [(df.iloc[i], score) for i, score in zip(valid_idx, scores)],
        key=lambda x: x[1],
        reverse=True
    )

    plats    = [r for (r, s) in items if r["type_plat"] == "plat"]
    desserts = [r for (r, s) in items if r["type_plat"] == "dessert"]
    entrees  = [r for (r, s) in items if r["type_plat"] == "entree"]

    packets = []

    while plats and (
        (N == 1)
        or (N == 2 and entrees)
        or (N >= 3 and entrees and desserts)
    ):
        packet = []

        if N == 1:
            packet.append(plats.pop(0))

        elif N == 2:
            packet.append(plats.pop(0))
            packet.append(entrees.pop(0))

        else:  # N >= 3
            packet.append(plats.pop(0))
            packet.append(desserts.pop(0))
            for _ in range(N - 2):
                if entrees:
                    packet.append(entrees.pop(0))

        packets.append(packet)

    return packets
