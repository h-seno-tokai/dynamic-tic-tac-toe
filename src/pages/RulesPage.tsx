import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/primitives';

export const RulesPage = () => {
  const navigate = useNavigate();

  return (
    <main className="mx-auto min-h-screen w-full max-w-3xl px-5 py-6">
      <header className="mb-8">
        <p className="text-sm font-medium text-accent">Rules</p>
        <h1 className="mt-2 text-3xl font-bold tracking-normal">ルール</h1>
      </header>

      <section className="grid gap-4 text-sm leading-7 text-muted">
        <p>自分の駒が、見えている状態で縦・横・斜めのいずれか一列に並ぶと勝利です。</p>
        <p>
          大きい駒は小さい駒の上に重ねられます。同じサイズやより大きい駒の上には置けません。
          自分の駒の上にも重ねられます。
        </p>
        <p>
          手駒から置くか、盤上で見えている自分の駒を別のマスへ移動します。3回同一局面が
          出現した場合や最大手数に達した場合は引き分けです。
        </p>
      </section>

      <div className="mt-8">
        <Button variant="secondary" onClick={() => navigate('/')}>
          メニューへ
        </Button>
      </div>
    </main>
  );
};
