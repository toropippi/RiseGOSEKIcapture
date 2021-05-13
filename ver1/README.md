# RiseGOSEKIcapture
 モンハンライズの護石を画像認識してテキスト化します。  
 護石が多すぎて管理しきれない、護石テーブルを自分で確かめたいけど手動でスキル内容を打ち込むのは面倒・・・という方に需要があるかと思い作りました。(需要があるかは知りません)  
 #モンハンライズ #MHR #MHRise #護石 #護石認識 #文字認識 #画像認識 #自動化 #手動 #テキスト化
## 使い方
　
### GOSEKIcapture.exeの起動
　GOSEKIcapture.exeを起動します。  
 <img width="711" alt="0" src="https://user-images.githubusercontent.com/44022497/116805810-ce870800-ab63-11eb-84f9-0acf59be8de2.png">  
 すると白いウィンドウの他に緑色の枠がでます。これもウィンドウなのでマウスドラッグで移動できます。  
 ![mouse](https://user-images.githubusercontent.com/44022497/116805822-e65e8c00-ab63-11eb-8465-e4d488d7f9c8.png)  

### キャプチャシーンに合わせる
 緑の枠をプレイ動画などのキャプチャシーンに合わせます。  
 このときプレイ画面は1920×1080でないと枠に合いません。Nintendo Switchの録画ボタンで録画した動画の場合1280×720なので、手動で動画サイズを引き伸ばすなりして下さい。  
 なるべく1920×1080に近づけて、枠も護石内容の文字がおさまるようにしてください。ここがアバウトだと認識精度に影響します。  
 ![1](https://user-images.githubusercontent.com/44022497/116805853-2b82be00-ab64-11eb-9223-5188be1233b8.jpg)

### スライドバーを移動させてカットオフ値を決める
 スライドバーを左右に移動させて適切なカットオフ値にして下さい。  
 下の画像のようにカットオフ値がMaxの場合、認識性能は上がりますが計算が重くなり、キャプチャフレームレートに影響します。  
 <img width="481" alt="3" src="https://user-images.githubusercontent.com/44022497/116805872-50773100-ab64-11eb-89cc-65bb2ec12a8d.png">  
 認識しないよりはマシなので基本カットオフ値Maxで良いと思います。護石のスキル候補が下に表示されます。  
 スキル名と、そのスキルの可能性が数値で表されます。数値は低いほど可能性が高いという意味です。  
 例えば「水属性攻撃強化」と「氷属性攻撃強化」は文字が非常によく似ているのでどちらも候補に出現することがありますが、より可能性の高いほうが採用されているのが数値を見て分かったりするので、眺めているだけでも楽しいです(人によります)。

### mesbox(テキスト画面)のOnOff
 「←」と書いてあるボタンを押すとmesbox(テキスト画面)が消えます。  
 テキスト内容に変更があったら常に更新をかけるので地味にフレームレートに影響することがあるので一応つけました。  
 もちろんテキスト画面Offにしても護石認識は機能します。  
 <img width="221" alt="4" src="https://user-images.githubusercontent.com/44022497/116805874-51a85e00-ab64-11eb-868a-b722e37ae34d.png">  

### コピペ
 <img width="481" alt="5" src="https://user-images.githubusercontent.com/44022497/116805875-5240f480-ab64-11eb-9adc-5368d032895d.png">  
 <img width="584" alt="6" src="https://user-images.githubusercontent.com/44022497/116805876-5240f480-ab64-11eb-8266-75ff9da4b801.png">  
  
テキスト出力はTab区切りで  
「スキル名1,Lv,スキル名2,Lv,スロA,スロB,スロC,護石のRARE度」  
となっています。  

## 注意事項
  
### 文字化け
 約1000個以上の護石をリストアップすると、文字数が32768(2の15乗)を超えて正しく表示されなくなるバグがあります。  
 ![mojibake](https://user-images.githubusercontent.com/44022497/116907491-9de6c180-ac7c-11eb-9095-3a4ab1ddae65.png)  
 これは表示の問題であり、ソフトの内部では正しい文字列で記憶されています。Escキーを押してソフトを終了することにより正しい文字列が「テキスト出力.txt」という名前で保存されます。  
 
### UIがはみ出てる
 ![juhuku](https://user-images.githubusercontent.com/44022497/116908342-bc00f180-ac7d-11eb-8df5-df91b7788260.png)  
 これは正しく表示されているときですが、「重複を数える」というチェックボックスが正しく表示されない場合があります。  
 その場合ディスプレイの表示スケールの設定を100%に変更して下さい。  
 ![kakusyuku](https://user-images.githubusercontent.com/44022497/116908436-ddfa7400-ac7d-11eb-8a3b-601a162182af.jpg)  
 これが150%とか200%だと微妙にUIが崩れてしまうことはありますが、ソフトの機能自体に不具合が起こることはありません。  
  
## スキル文字認識の原理
 テンプレートマッチングみたいなことをやってます。  
 そのうちディープラーニングで認識精度と速度の両立を図りたい・・・  
 
## 更新履歴
2021/5/3 ver 1.11 重複を数えない設定を追加  
2021/5/3 ver 1.10 達人芸を対応。出力文字数が32767を超えるとバグるのを修正  
2021/5/2 ver 1.0 公開  