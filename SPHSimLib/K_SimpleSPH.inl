# i f n d e f   _ _ K _ S i m p l e S P H _ c u _ _  
 # d e f i n e   _ _ K _ S i m p l e S P H _ c u _ _  
  
 # i n c l u d e   " K _ C o m m o n . c u h "  
 # i n c l u d e   " c u t i l _ m a t h . h "  
 # i n c l u d e   " v e c t o r _ t y p e s . h "  
  
 u s i n g   n a m e s p a c e   S i m L i b ;  
 u s i n g   n a m e s p a c e   S i m L i b : : S i m : : S i m p l e S P H ;  
  
 # i n c l u d e   " K _ U n i f o r m G r i d _ U t i l s . i n l "  
 # i n c l u d e   " K _ C o l o r i n g . i n l "  
 # i n c l u d e   " K _ S P H _ C o m m o n . c u h "  
  
 c l a s s   S i m p l e S P H S y s t e m  
 {  
 p u b l i c :  
  
 	 s t a t i c   _ _ d e v i c e _ _   v o i d   U p d a t e S o r t e d V a l u e s ( S i m p l e S P H D a t a   & d P a r t i c l e s S o r t e d ,   S i m p l e S P H D a t a   & d P a r t i c l e s ,   u i n t   & i n d e x ,   u i n t   & s o r t e d I n d e x )  
 	 {  
 	 	 d P a r t i c l e s S o r t e d . p o s i t i o n [ i n d e x ] 	 =   F E T C H _ N O T E X ( d P a r t i c l e s , p o s i t i o n , s o r t e d I n d e x ) ;  
 	 	 d P a r t i c l e s S o r t e d . v e l o c i t y [ i n d e x ] 	 =   F E T C H _ N O T E X ( d P a r t i c l e s , v e l o c i t y , s o r t e d I n d e x ) ;  
 	 	 d P a r t i c l e s S o r t e d . v e l e v a l [ i n d e x ] 	 	 =   F E T C H _ N O T E X ( d P a r t i c l e s , v e l e v a l , s o r t e d I n d e x ) ;  
 	 	 / / d P a r t i c l e s S o r t e d . c o l o r [ i n d e x ] 	 	 =   F E T C H _ N O T E X ( d P a r t i c l e s , c o l o r , s o r t e d I n d e x ) ;  
  
 	 	 / / d P a r t i c l e s S o r t e d . v e l e v a l _ d i f f [ i n d e x ] =   F E T C H _ N O T E X ( d P a r t i c l e s , v e l e v a l _ d i f f , s o r t e d I n d e x ) ;  
 	 	 / / d P a r t i c l e s S o r t e d . s p h _ f o r c e [ i n d e x ] 	 =   F E T C H _ N O T E X ( d P a r t i c l e s , s p h _ f o r c e , s o r t e d I n d e x ) ;  
 	 	 / / d P a r t i c l e s S o r t e d . p r e s s u r e [ i n d e x ] 	 =   F E T C H _ N O T E X ( d P a r t i c l e s , p r e s s u r e , s o r t e d I n d e x ) ;  
 	 	 / / d P a r t i c l e s S o r t e d . d e n s i t y [ i n d e x ] 	 	 =   F E T C H _ N O T E X ( d P a r t i c l e s , d e n s i t y , s o r t e d I n d e x ) ;  
 	 }  
 } ;  
  
  
 # i n c l u d e   " K _ S i m p l e S P H _ S t e p 1 . i n l "  
 # i n c l u d e   " K _ S i m p l e S P H _ S t e p 2 . i n l "  
 # i n c l u d e   " K _ S i m p l e S P H _ I n t e g r a t e . i n l "  
  
  
 # e n d i f  
 