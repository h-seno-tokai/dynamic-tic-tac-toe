import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/primitives';

const sections = [
  {
    title: '目的',
    body: '自分の駒が「見えている状態」で、縦・横・斜めのいずれか一列に並べば勝利です。駒が相手の駒に覆われて見えなくなっている場合は、一列に並んでいても勝利になりません。',
  },
  {
    title: '駒の種類',
    body: '各プレイヤーは大・中・小の3サイズの駒をそれぞれ複数枚ずつ持ちます。大きい駒ほど強く、小さい駒を「覆い被せる（ゴブル）」ことができます。同サイズや自分より大きい駒の上には置けません。',
  },
  {
    title: '手番にできること',
    body: '手番のプレイヤーは次のいずれか1つを行います。\n①手駒（まだ盤上にない駒）を盤上の任意のマスに置く。\n②盤上で「見えている」自分の駒を別のマスへ移動する。',
  },
  {
    title: '覆い被せのルール',
    body: '大きい駒は相手・自分を問わず、より小さい駒の上に置けます。覆われた駒は盤上に残りますが見えなくなります。覆っていた駒を別のマスへ移動させると、下の駒が再び現れ、そのプレイヤーのものとして機能します。',
  },
  {
    title: '注意：移動で相手を勝たせてしまう場合',
    body: '自分の駒を移動させたとき、その下にあった相手の駒が現れて相手の一列が完成してしまう場合でも、移動は取り消せません。ただし、移動した駒を相手の一列を崩す位置（その列の駒の上）に置ける場合はそこに置かなければなりません。',
  },
  {
    title: '引き分け',
    body: '同一局面が3回繰り返された場合、または最大手数に達した場合は引き分けです。',
  },
];

export const RulesPage = () => {
  const navigate = useNavigate();

  return (
    <main className="mx-auto min-h-screen w-full max-w-3xl px-5 py-6">
      <header className="mb-8">
        <p className="text-sm font-medium text-accent">Rules</p>
        <h1 className="mt-2 text-3xl font-bold tracking-normal">ルール</h1>
        <p className="mt-2 text-sm text-muted">Gobblet Gobblers をベースにしたボードゲームです。</p>
      </header>

      <div className="grid gap-6">
        {sections.map((s) => (
          <section key={s.title}>
            <h2 className="mb-2 text-base font-semibold">{s.title}</h2>
            <p className="text-sm leading-7">
              {s.body.split('\n').map((line, i) => (
                <span key={i}>
                  {line}
                  {i < s.body.split('\n').length - 1 && <br />}
                </span>
              ))}
            </p>
          </section>
        ))}
      </div>

      <div className="mt-10">
        <Button variant="secondary" onClick={() => navigate('/')}>
          メニューへ
        </Button>
      </div>
    </main>
  );
};
