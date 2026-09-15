"""Read-only review refuses corrupted archives and misleading rank populations."""
import hashlib,io,json,tarfile,tempfile,unittest
from pathlib import Path
import numpy as np
from research.relative_leadership_audit import unpack_verified, pair_accuracy

class ReviewTests(unittest.TestCase):
    def bundle(self, corrupt=False):
        output=io.BytesIO();payload=b'original account\n'
        manifest={'account.txt':hashlib.sha256(payload).hexdigest()}
        with tarfile.open(fileobj=output,mode='w:gz') as t:
            for name,body in [('nonlinear/account.txt',payload+b'bad' if corrupt else payload),
                              ('nonlinear/MANIFEST.json',json.dumps(manifest).encode())]:
                info=tarfile.TarInfo(name);info.size=len(body);t.addfile(info,io.BytesIO(body))
        return output.getvalue()
    def test_exact_archive_and_members(self):
        b=self.bundle()
        with tempfile.TemporaryDirectory() as tmp:
            root,count=unpack_verified(b,hashlib.sha256(b).hexdigest(),Path(tmp))
            self.assertEqual(count,1);self.assertEqual((root/'account.txt').read_bytes(),b'original account\n')
    def test_manifest_corruption_is_rejected(self):
        b=self.bundle(corrupt=True)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):unpack_verified(b,hashlib.sha256(b).hexdigest(),Path(tmp))
    def test_outer_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):unpack_verified(self.bundle(),'0'*64,Path(tmp))
    def test_pair_ties_and_empty_population(self):
        self.assertEqual(pair_accuracy([3,2,1],[3,2,1]),1.)
        self.assertEqual(pair_accuracy([1,2,3],[3,2,1]),0.)
        self.assertEqual(pair_accuracy([1,1,1],[3,2,1]),.5)
        self.assertIsNone(pair_accuracy([1,2,3],[1,1,1]))
        with self.assertRaises(ValueError):pair_accuracy([1,np.nan],[1,2])

if __name__=='__main__':unittest.main()
